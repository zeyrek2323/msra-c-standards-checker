"""
Configuration file for Llama3 LoRA fine-tuning
"""

import os

class TrainingConfig:
    """Training configuration class"""
    
    # Model Configuration
    BASE_MODEL_NAME =r"C:\Users\VICTUS\Meta-Llama-3-8B"  # Change this to your Llama3 model
    DATASET_PATH = "dataset_msra_instruct.jsonl"
    OUTPUT_DIR = "./llama3-lora-msra"
    
    # LoRA Configuration
    LORA_R = 16  # Rank
    LORA_ALPHA = 32  # Alpha scaling
    LORA_DROPOUT = 0.1
    # Reduce target modules to minimize VRAM footprint
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]
    
    # Training Configuration
    NUM_EPOCHS = 1
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION_STEPS = 8
    LEARNING_RATE = 2e-4
    WARMUP_STEPS = 100
    MAX_LENGTH = 256
    MAX_NEW_TOKENS = 200
    
    # Quantization Configuration
    LOAD_IN_4BIT = True
    BNB_4BIT_USE_DOUBLE_QUANT = True
    BNB_4BIT_QUANT_TYPE = "nf4"
    # Use float16 for lower VRAM pressure
    BNB_4BIT_COMPUTE_DTYPE = "float16"
    
    # Logging and Saving
    LOGGING_STEPS = 10
    SAVE_STEPS = 500
    EVAL_STEPS = 500
    REPORT_TO_WANDB = True
    RUN_NAME = "llama3-lora-msra"
    
    # Hardware Configuration
    FP16 = True
    DEVICE_MAP = "auto"
    
    @classmethod
    def get_lora_config(cls):
        """Get LoRA configuration"""
        from peft import LoraConfig, TaskType
        
        return LoraConfig(
            r=cls.LORA_R,
            lora_alpha=cls.LORA_ALPHA,
            target_modules=cls.LORA_TARGET_MODULES,
            lora_dropout=cls.LORA_DROPOUT,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    
    @classmethod
    def get_bnb_config(cls):
        """Get BitsAndBytes configuration"""
        from transformers import BitsAndBytesConfig
        import torch
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map.get(str(cls.BNB_4BIT_COMPUTE_DTYPE).lower(), torch.float16)
        
        return BitsAndBytesConfig(
            load_in_4bit=cls.LOAD_IN_4BIT,
            bnb_4bit_use_double_quant=cls.BNB_4BIT_USE_DOUBLE_QUANT,
            bnb_4bit_quant_type=cls.BNB_4BIT_QUANT_TYPE,
            bnb_4bit_compute_dtype=compute_dtype
        )
    
    @classmethod
    def get_training_args(cls, force_cpu: bool = False):
        """Get training arguments"""
        from transformers import TrainingArguments
        
        # Normalize report_to to expected type across versions
        report_to = ["wandb"] if cls.REPORT_TO_WANDB else ["none"]
        
        kwargs = dict(
            output_dir=cls.OUTPUT_DIR,
            num_train_epochs=cls.NUM_EPOCHS,
            per_device_train_batch_size=cls.BATCH_SIZE,
            per_device_eval_batch_size=cls.BATCH_SIZE,
            gradient_accumulation_steps=cls.GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=cls.WARMUP_STEPS,
            learning_rate=cls.LEARNING_RATE,
            fp16=False if force_cpu else cls.FP16,
            logging_steps=cls.LOGGING_STEPS,
            save_steps=cls.SAVE_STEPS,
            eval_steps=cls.EVAL_STEPS,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=report_to,
            run_name=cls.RUN_NAME,
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )
        if force_cpu:
            kwargs["no_cuda"] = True
        
        try:
            return TrainingArguments(**kwargs)
        except TypeError:
            # Fallback for Transformers versions where 'evaluation_strategy' was renamed
            kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
            return TrainingArguments(**kwargs)
