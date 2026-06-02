import numpy as np 
import pandas as pd
import csv
import nltk
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer,GPT2Model, GPT2Config, AdamW
from datasets import load_dataset, concatenate_datasets
from tqdm import tqdm
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

class SoftPromptTuning(torch.nn.Module):
    def __init__(self, model, num_soft_tokens=3, prompt_text="summarize"):
        super(SoftPromptTuning, self).__init__()
        self.model = model
        self.num_soft_tokens = num_soft_tokens
        
        prompt_input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

        with torch.no_grad():
            prompt_embeddings = model.transformer.wte(prompt_input_ids).squeeze(0)
            
        self.soft_prompt = torch.nn.Embedding(num_soft_tokens, model.config.n_embd)

        self.soft_prompt.weight.data.copy_(prompt_embeddings[:num_soft_tokens])
        
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):

        batch_size = input_ids.size(0)
        soft_prompt_tokens = self.soft_prompt.weight.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = self.model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([soft_prompt_tokens, inputs_embeds], dim=1)
        
        if attention_mask is not None:
            soft_attention_mask = torch.ones(batch_size, self.num_soft_tokens).to(attention_mask.device)
            attention_mask = torch.cat([soft_attention_mask, attention_mask], dim=1)
        
        if labels is not None:
            soft_label_padding = torch.full((batch_size, self.num_soft_tokens), -100).to(labels.device)
            labels = torch.cat([soft_label_padding, labels], dim=1)
            
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)

    def generate(self, input_ids, attention_mask=None, max_new_tokens=128, pad_token_id=None, num_beams=4, early_stopping=True):
        batch_size = input_ids.size(0)
        soft_prompt_tokens = self.soft_prompt.weight.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = self.model.transformer.wte(input_ids)
        inputs_embeds = torch.cat([soft_prompt_tokens, inputs_embeds], dim=1)
        
        if attention_mask is not None:
            soft_attention_mask = torch.ones(batch_size, self.num_soft_tokens).to(attention_mask.device)
            attention_mask = torch.cat([soft_attention_mask, attention_mask], dim=1)
        
        outputs = self.model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=max_new_tokens, pad_token_id=pad_token_id, num_beams=num_beams, early_stopping=True)
        return outputs

soft_prompt_model = SoftPromptTuning(model)

device = "cuda" if torch.cuda.is_available() else "cpu"

soft_prompt_model = soft_prompt_model.to(device)

def custom_collate_fn(batch, max_input_length = 256, max_output_length = 128):
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
    collate_fn=custom_collate_fn
)

val_dataloader = DataLoader(
    validation_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=custom_collate_fn
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

train_model(soft_prompt_model)

torch.save(soft_prompt_model.state_dict(), "soft_prompt_model.pt")

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

def compute_rouge(model, dataloader, num_beams=4):
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

test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)

evaluate_loss(soft_prompt_model, val_dataloader)
finetune_results = compute_rouge(soft_prompt_model, test_dataloader)
print("Traditional Finetuning Results:", finetune_results)