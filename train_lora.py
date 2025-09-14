#!/usr/bin/env python3
"""
Llama3 LoRA Fine-tuning Script for MSRA C Standard Dataset
"""

import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training
)
import wandb
from tqdm import tqdm

from config import TrainingConfig

def load_dataset(file_path):
    """Load and format the MSRA dataset"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                # Format for instruction following
                prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n"
                completion = item['output']
                
                data.append({
                    "text": prompt + completion,
                    "prompt": prompt,
                    "completion": completion
                })
    
    return Dataset.from_list(data)

def main():
    # Configuration
    config = TrainingConfig()
    
    print("Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL_NAME,
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Build a conservative max_memory map to enable GPU/CPU offload if needed
    max_memory = None
    try:
        if torch.cuda.is_available():
            total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
            # Leave ~1GB headroom
            gpu_budget_gib = max(int((total_vram_bytes - 1_000_000_000) / (1024**3)), 1)
            import psutil  # type: ignore
            total_ram_bytes = psutil.virtual_memory().total
            cpu_budget_gib = max(int((total_ram_bytes - 4_000_000_000) / (1024**3)), 4)
            max_memory = {"cuda:0": f"{gpu_budget_gib}GiB", "cpu": f"{cpu_budget_gib}GiB"}
    except Exception:
        pass

    # Load model with quantization and potential offload
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME,
            quantization_config=config.get_bnb_config(),
            device_map=config.DEVICE_MAP,
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    except ValueError as e:
        print(f"Primary load failed ({e}). Falling back to CPU-only loading...")
        # Fallback to CPU-only to avoid quantizer validation error
        model = AutoModelForCausalLM.from_pretrained(
            config.BASE_MODEL_NAME,
            quantization_config=config.get_bnb_config(),
            device_map={"": "cpu"},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
    
    # Prepare model for training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    model = get_peft_model(model, config.get_lora_config())
    model.print_trainable_parameters()
    
    print("Loading dataset...")
    
    # Load and format dataset
    dataset = load_dataset(config.DATASET_PATH)
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LENGTH,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=config.get_training_args(),
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(min(100, len(tokenized_dataset)))),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting training...")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    print(f"Model saved to {config.OUTPUT_DIR}")
    
    # Save LoRA adapter separately
    model.save_pretrained(f"{config.OUTPUT_DIR}/lora_adapter")
    print("LoRA adapter saved separately")

if __name__ == "__main__":
    main()
