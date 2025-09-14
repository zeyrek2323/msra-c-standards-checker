#!/usr/bin/env python3
"""
Inference script for the fine-tuned Llama3 LoRA model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

def load_model_and_tokenizer(base_model_path, lora_adapter_path):
    """Load the base model and LoRA adapter"""
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        lora_adapter_path,
        is_trainable=False
    )
    
    # Set eval mode for faster inference
    model.eval()
    
    return model, tokenizer


def generate_response(model, tokenizer, instruction, input_code, max_length=512):
    """Generate response using the fine-tuned model"""
    
    # Create more structured prompt
    prompt = f"""### Instruction:
{instruction}

### Input:
{input_code}

### Response:
"""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Inform user we're generating
    print("Generating response...", flush=True)
    
    # Generate response with faster, deterministic settings
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    generated_text = response[len(prompt):]
    
    return generated_text.strip()

def main():
    # Configuration
    base_model_path = r"C:\Users\VICTUS\Meta-Llama-3-8B"  # Local Llama3 base model
    lora_adapter_path = "./llama3-lora-msra/lora_adapter"
    
    print("Loading model and tokenizer...")
    
    try:
        model, tokenizer = load_model_and_tokenizer(base_model_path, lora_adapter_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Interactive mode
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("="*50)
    print("Type 'quit' to exit")
    
    while True:
        try:
            instruction = input("\nEnter instruction (or 'quit'): ").strip()
            if instruction.lower() == 'quit':
                break
            
            input_code = input("Enter C code: ").strip()
            
            if not input_code:
                print("Please enter valid C code")
                continue
            
            response = generate_response(model, tokenizer, instruction, input_code)
            print(f"\nResponse: {response}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
