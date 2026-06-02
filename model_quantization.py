import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quantize_weights(weights, bit_precision=8):
    max_val = weights.abs().max()
    scale = (2 ** (bit_precision - 1) - 1) / max_val
    quantized_weights = (weights * scale).round().clamp(-2 ** (bit_precision - 1), 2 ** (bit_precision - 1) - 1).to(torch.int8)
    return quantized_weights / scale
    
def apply_whole_model_quantization(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = quantize_weights(param.data)
    return model
    
def apply_selective_component_quantization(model):
    for name, param in model.named_parameters():
        if 'attn' in name or 'ffn' in name:
            if param.requires_grad:
                param.data = quantize_weights(param.data)
    return model

test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:3000]")

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id

def memory_usage(model):
    return model.get_memory_footprint() / (1024 ** 2)

def measure_latency(model, tokenizer, dataset):
    model.eval()
    total_time = 0
    total_samples = 0
    for example in dataset:
        inputs = tokenizer(example['text'], return_tensors="pt", truncation=True, max_length=512).to(model.device)
        if inputs['input_ids'].numel() == 0:
            continue
        start_time = time.time()
        with torch.no_grad():
            _ = model(**inputs)
            
        end_time = time.time()
        total_time += end_time - start_time
        total_samples += 1
    
    return total_time / total_samples

def compute_perplexity(model, tokenizer, dataset):
    """Calculate average perplexity over a dataset."""
    model.eval()
    total_loss = 0
    total_examples = 0

    for example in dataset:
        inputs = tokenizer(example['text'], return_tensors="pt", truncation=True, max_length=512).to(model.device)
        if inputs['input_ids'].numel() == 0:
            continue
        
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs['input_ids'])
            total_loss += outputs.loss.item() * inputs['input_ids'].size(1)
            total_examples += inputs['input_ids'].size(1)
    
    avg_loss = total_loss / total_examples
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

quantized_model = apply_whole_model_quantization(model)
selectively_quantized_model = apply_selective_component_quantization(model)

initial_memory = memory_usage(model)
baseline_latency = measure_latency(model, tokenizer, test_dataset)
baseline_perplexity = compute_perplexity(model, tokenizer, test_dataset)
print(f"Baseline - Memory: {initial_memory} MB, Latency: {baseline_latency:.4f}s, Perplexity: {baseline_perplexity:.4f}")

quantized_memory = memory_usage(quantized_model)
quantized_latency = measure_latency(quantized_model, tokenizer, test_dataset)
quantized_perplexity = compute_perplexity(quantized_model, tokenizer, test_dataset)
print(f"Whole-Model Quantization - Memory: {quantized_memory} MB, Latency: {quantized_latency:.4f}s, Perplexity: {quantized_perplexity:.4f}")

selective_memory = memory_usage(selectively_quantized_model)
selective_latency = measure_latency(selectively_quantized_model, tokenizer, test_dataset)
selective_perplexity = compute_perplexity(selectively_quantized_model, tokenizer, test_dataset)
print(f"Selective Quantization - Memory: {selective_memory} MB, Latency: {selective_latency:.4f}s, Perplexity: {selective_perplexity:.4f}")

config_8bit = BitsAndBytesConfig(load_in_8bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=config_8bit,
    device_map='auto'
)

config_4bit = BitsAndBytesConfig(load_in_4bit=True)

model_4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=config_4bit,
    device_map='auto'
)

config_nf4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
)

model_nf4bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=config_nf4bit,
    device_map='auto'
)

def evaluate_metrics(model, test_dataset, tokenizer):
    memory = memory_usage(model)
    latency = measure_latency(model, tokenizer, test_dataset)
    perplexity = compute_perplexity(model, tokenizer, test_dataset)

    print(f"Memory Usage (MB): {memory:.2f}")
    print(f"Inference Latency (s): {latency:.4f}")
    print(f"Perplexity: {perplexity:.2f}")

print("Evaluating 8-bit Quantization:")
evaluate_metrics(model_8bit, test_dataset, tokenizer)

print("\nEvaluating 4-bit Quantization:")
evaluate_metrics(model_4bit, test_dataset, tokenizer)

print("\nEvaluating NF4 Quantization:")
evaluate_metrics(model_nf4bit, test_dataset, tokenizer)