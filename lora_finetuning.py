import numpy as np 
import pandas as pd
import csv
import nltk
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer,GPT2Model, GPT2Config, AdamW
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets
from peft import get_peft_model, LoraConfig
import time
import evaluate

# Load ROUGE metric
rouge = evaluate.load("rouge")

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset('csv', data_files={
    'train': '/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/train.csv',
    'test': '/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/test.csv',
    'validation': '/kaggle/input/newspaper-text-summarization-cnn-dailymail/cnn_dailymail/validation.csv'})

combined_dataset = concatenate_datasets([dataset['train'], dataset['test'], dataset['validation']])

combined_dataset = combined_dataset.shuffle(seed=42)  # Seed for reproducibility

train_size = 21000
validation_size = 6000
test_size = 3000

train_dataset = combined_dataset.select(range(train_size))
validation_dataset = combined_dataset.select(range(train_size, train_size + validation_size))
test_dataset = combined_dataset.select(range(train_size + validation_size, train_size + validation_size + test_size))

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["attn.c_attn"],
    lora_dropout=0.1,
    bias="none"
)

lora_model = get_peft_model(model, lora_config)

for name, param in lora_model.named_parameters():
    if 'lora' not in name:
        param.requires_grad = False

device = "cuda" if torch.cuda.is_available() else "cpu"

def collate_fn(batch, max_input_length = 256, max_output_length = 128):

    articles = [item['article'] for item in batch]
    summaries = [item['highlights'] for item in batch]
    
    inputs = tokenizer(articles, max_length = max_input_length, truncation=True, padding='max_length', return_tensors="pt")
    labels = tokenizer(summaries, max_length = max_output_length, truncation=True, padding='max_length', return_tensors="pt")["input_ids"]
    
    labels_padded = torch.full((labels.size(0), max_input_length), -100, dtype=torch.long)
    labels_padded[:, :labels.size(1)] = labels
    
    inputs["labels"] = labels_padded

    return inputs

train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataloader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn
)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def train_model(model, epochs=3, lr=5e-5):
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    start_time = time.time()
    
    for epoch in range(epochs):
        best_val_loss = float('inf')
        
        model.train()
        running_loss = 0.0
        for batch in tqdm(train_dataloader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device) 
            optimizer.zero_grad()
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}, Train Loss: {running_loss/len(train_dataloader)}")
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                inputs = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
        else:
            print("Early stopping triggered.")
            break
            
    end_time = time.time()

    total_training_time = end_time - start_time
    print(f"Total training time: {total_training_time:.2f} seconds")
    
    gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
    print(f"GPU Memory Usage after training: {gpu_memory_used:.2f} MB")

train_model(lora_model)

torch.save(lora_model.state_dict(), "lora_finetuned_model.pt")

def evaluate_loss(model, dataloader):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss}")

def compute_rouge(model, dataloader):
    model.eval()
    predictions, references = [], []
    
    tokenizer.padding_side = 'left'
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            generated_outputs = model.generate(input_ids=inputs, attention_mask=attention_mask, max_new_tokens=128, pad_token_id=tokenizer.pad_token_id, num_beams=4, early_stopping=True)
            decoded_preds = tokenizer.batch_decode(generated_outputs, skip_special_tokens=True)
            labels = torch.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]
            
            predictions.extend(decoded_preds)
            references.extend(decoded_labels)
            
    tokenizer.padding_side = 'right'            
            
    result = rouge.compute(predictions=predictions, references=references)
    return result

test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

evaluate_loss(lora_model, val_dataloader)
lora_results = compute_rouge(lora_model, test_dataloader)
print("LoRA Results:", lora_results)