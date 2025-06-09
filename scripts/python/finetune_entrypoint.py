import argparse
import json
import os
import torch
import sys
import bitsandbytes as bnb
from accelerate import Accelerator
from datasets import load_dataset # load_dataset can load local jsonl
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from huggingface_hub import login
import mlflow
from mlflow.models import infer_signature
import pandas as pd # For signature example

# Add utils directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, 'utils')
if utils_dir not in sys.path:
    sys.path.insert(0, utils_dir)

# Now import from utils
try:
    from mlflow_callbacks import create_mlflow_callback
    print("Successfully imported MLflow callback")
except ImportError as e:
    print(f"Warning: Could not import MLflow callback: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Utils directory: {utils_dir}")
    print(f"Utils directory exists: {os.path.exists(utils_dir)}")
    if os.path.exists(utils_dir):
        print(f"Contents of utils directory: {os.listdir(utils_dir)}")
    # Define a dummy callback as fallback
    def create_mlflow_callback(log_every_n_steps=5):
        print("Using dummy callback - MLflow progressive logging disabled")
        return None

# LLAMA_3_CHAT_TEMPLATE (Ensure this is the one you want to use)
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] + eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] + eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

def create_training_arguments(args):
    """Create training arguments with progressive logging enabled"""
    output_dir_trainer = "/tmp/trainer_outputs"
    
    return TrainingArguments(
        output_dir=output_dir_trainer,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        
        # UPDATED: Enable progressive logging and evaluation
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        
        save_strategy="steps", 
        save_steps=args.eval_steps * 2,  # Save less frequently than eval
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        
        # Precision and optimization
        bf16=args.bf16,
        tf32=args.bf16,
        
        # MLflow integration
        report_to="none",  # We'll handle MLflow with our custom callback
        
        # Other settings
        seed=args.seed,
        max_grad_norm=1.0,
        warmup_steps=max(1, args.logging_steps // 2),  # Adaptive warmup
        lr_scheduler_type="cosine",
        
        # Memory optimizations
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        eval_accumulation_steps=1,
        prediction_loss_only=True,
    )
    
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(hf_model):
    lora_module_names = set()
    for name, module in hf_model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit): # or bnb.nn.Linear8bitLt for 8-bit
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def parse_arguments():
    parser = argparse.ArgumentParser()
    # --- Arguments passed from finetune_classifier_hf.py ---
    # Model and Data
    parser.add_argument("--model_id", type=str, required=True, help="Hugging Face model ID.")
    parser.add_argument("--train_file_name", type=str, required=True, help="Name of the training data file (e.g., train_dataset.jsonl).")
    parser.add_argument("--eval_file_name", type=str, required=True, help="Name of the evaluation data file (e.g., validation_dataset.jsonl).")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--hf_token", type=str, default=None)

    # Training Hyperparameters (from train_config in caller)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    # parser.add_argument("--lr", type=float, default=2e-4, help="Renamed from 'lr' to 'learning_rate' for consistency.")
    parser.add_argument("--lr", type=float, default=2e-4) # Keeping 'learning_rate' as per finetune_classifier_hf.py
    parser.add_argument("--bf16", type=str_to_bool, default=True)
    parser.add_argument("--gradient_checkpointing", type=str_to_bool, default=True)
    parser.add_argument("--logging_steps", type=int, default=5)
    # Additional common training args, can be added if needed & passed from caller
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4) # Default from your original script
    parser.add_argument("--seed", type=int, default=42) # Added for reproducibility

    # LoRA Hyperparameters (from lora_config in caller)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj", help="Comma-separated list (e.g., q_proj,v_proj).")
    parser.add_argument("--merge_weights", type=str_to_bool, default=True)

    # MLflow arguments (passed from caller)
    parser.add_argument("--mlflow_arn", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None, help="Parent MLflow run ID.")

    # SageMaker specific (usually not needed as explicit args if using env vars, but good for clarity if passed)
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--eval_data_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"))
    # parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")) # For non-model outputs
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model")) # Final model output

    # arguments for better MLFLow logging:
    parser.add_argument("--limit_train_samples", type=int, default=None)
    parser.add_argument("--limit_eval_samples", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=None)  # NEW: Evaluation frequency

    args = parser.parse_args()

    # Set eval_steps if not provided
    if args.eval_steps is None:
        args.eval_steps = args.logging_steps * 2  # Evaluate every 2x logging frequency
    
    return args

def str_to_bool(value):
    if isinstance(value, bool): return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    args = parse_arguments()

    if args.hf_token:
        print(f"Logging into Hugging Face Hub with token.")
        login(token=args.hf_token)
        
    print("############################################")
    print("Number of GPUs: ", torch.cuda.device_count())
    print("############################################")
    
    accelerator = Accelerator()

    # --- 1. Load Tokenizer ---
    print(f"Loading tokenizer for: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_auth_token=bool(args.hf_token))
    tokenizer.pad_token = tokenizer.eos_token # Important for padding
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE # Set your desired chat template

    # --- 2. Load and Prepare Datasets ---
    train_file_path = os.path.join(args.train_data_dir, args.train_file_name)
    eval_file_path = os.path.join(args.eval_data_dir, args.eval_file_name)

    print(f"Loading training data from: {train_file_path}")
    print(f"Loading evaluation data from: {eval_file_path}")

    try:
        # load_dataset is robust for jsonl
        train_dataset_raw = load_dataset("json", data_files=train_file_path, split="train")
        eval_dataset_raw = load_dataset("json", data_files=eval_file_path, split="train")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise

    # Template and tokenize datasets
    def template_and_tokenize_fn(examples):
        # Apply chat template (ensure 'messages' field exists in your JSONL)
        templated_texts = tokenizer.apply_chat_template(
            examples["messages"], tokenize=False, add_generation_prompt=False
        )
        # Tokenize
        return tokenizer(templated_texts, truncation=True, max_length=min(args.max_seq_length, 512), padding="max_length")

    with accelerator.main_process_first(): # Ensures download/mapping happens once
        tokenized_train_dataset = train_dataset_raw.map(
            template_and_tokenize_fn,
            batched=True,
            remove_columns=train_dataset_raw.column_names # Remove original columns
        )
        tokenized_eval_dataset = eval_dataset_raw.map(
            template_and_tokenize_fn,
            batched=True,
            remove_columns=eval_dataset_raw.column_names
        )

    # Limit dataset sizes if specified (for faster testing)
    # if args.limit_train_samples:
    #     tokenized_train_dataset = tokenized_train_dataset.select(
    #         range(min(args.limit_train_samples, len(tokenized_train_dataset)))
    #     )
    #     print(f"Limited training dataset to {len(tokenized_train_dataset)} samples")
    
    # if args.limit_eval_samples:
    #     tokenized_eval_dataset = tokenized_eval_dataset.select(
    #         range(min(args.limit_eval_samples, len(tokenized_eval_dataset)))
    #     )
    #     print(f"Limited eval dataset to {len(tokenized_eval_dataset)} samples")

    print(f"Processed train samples: {len(tokenized_train_dataset)}")
    print(f"Processed eval samples: {len(tokenized_eval_dataset)}")
    if len(tokenized_train_dataset) > 0: print("Sample train input_ids:", tokenized_train_dataset[0]['input_ids'][:20])


    # --- 3. Setup Model (QLoRA) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 # Ensure your GPU supports bfloat16
    )

    print(f"Loading base model ({args.model_id}) with QLoRA config...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={'': torch.cuda.current_device()}, # Use accelerator's device
        trust_remote_code=True, # If required by the model
        use_auth_token=bool(args.hf_token),
        cache_dir="/tmp/.cache" # Writable cache
    )
    print("Base model loaded.")

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    # Determine LoRA target modules
    if args.lora_target_modules and args.lora_target_modules.lower() != "auto":
        modules = args.lora_target_modules.split(',')
    else:
        print("Finding all linear names for LoRA...")
        modules = find_all_linear_names(model)
    print(f"LoRA target modules: {modules}")

    lora_config_obj = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config_obj)
    print_trainable_parameters(model)
    print("PEFT model created.")

    # --- 4. Setup Trainer ---
    training_args_obj = create_training_arguments(args)

    # Add MLflow callback
    callbacks = []
    if args.mlflow_arn:
        mlflow_callback = create_mlflow_callback(log_every_n_steps=args.logging_steps)
        callbacks.append(mlflow_callback)
        print(f"Added MLflow callback - will log every {args.logging_steps} steps")

    trainer = Trainer(
        model=model, # PEFT model
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        args=training_args_obj,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=callbacks,
    )
    model.config.use_cache = False # Recommended for training

    # --- 5. Train and Save ---
    # MLflow setup: use the parent run_id passed from the pipeline step
    if args.mlflow_arn and args.experiment_name and args.run_id:
        mlflow.set_tracking_uri(args.mlflow_arn)
        mlflow.set_experiment(args.experiment_name)
        # Start a nested run or continue the parent run.
        # Using nested=True if the Trainer also reports to MLflow with its own run_name.
        # Or, if Trainer's report_to is "none", this becomes the primary MLflow context for this script.
        with mlflow.start_run(run_id=args.run_id, nested=True) as current_run: # Use parent run_id
            mlflow.log_param("script_model_id", args.model_id)
            mlflow.log_params({ # Log key training parameters
                "script_epochs": args.epochs, "script_learning_rate": args.lr,
                "script_batch_size": args.per_device_train_batch_size,
                "script_lora_r": args.lora_r, "script_lora_alpha": args.lora_alpha,
                "script_target_modules": modules
            })
            print(f"MLflow: Logging to existing run ID: {args.run_id}, experiment: {args.experiment_name}")

            print("Starting model training...")
            train_result = trainer.train()
            print("Training finished.")
            # trainer.log_metrics("train", train_result.metrics) # Trainer automatically logs if report_to="mlflow"
            # trainer.save_metrics("train", train_result.metrics)


            # The best model (PEFT adapter) is already loaded if load_best_model_at_end=True
            best_peft_model = trainer.model
            
            # Final model save directory (SageMaker standard)
            final_model_output_dir = args.model_dir
            os.makedirs(final_model_output_dir, exist_ok=True)

            if args.merge_weights:
                print("Merging LoRA weights with base model...")
                # Save PEFT adapter temporarily to reload for merging
                temp_adapter_dir = "/tmp/temp_peft_adapter_for_merge"
                best_peft_model.save_pretrained(temp_adapter_dir, safe_serialization=True) # safe_serialization for PEFT can be True

                del model, trainer, best_peft_model # Free up VRAM
                torch.cuda.empty_cache()
                accelerator.free_memory()


                print("Reloading base model for merging...")
                # Reload base model in a higher precision (e.g., bf16) for stable merging
                base_model_for_merge = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                    device_map="auto", # Let transformers handle device placement for merging
                    trust_remote_code=True,
                    use_auth_token=bool(args.hf_token),
                    cache_dir="/tmp/.cache"
                )
                
                print(f"Loading PEFT adapter from {temp_adapter_dir} and merging...")
                # Load the PEFT model using the base model and the adapter
                merged_model = get_peft_model(base_model_for_merge, lora_config_obj) # Re-apply config
                merged_model.load_adapter(temp_adapter_dir, adapter_name="default") # Load the saved best adapter
                merged_model = merged_model.merge_and_unload() # Merge
                
                print(f"Saving merged model to {final_model_output_dir}")
                merged_model.save_pretrained(final_model_output_dir, safe_serialization=True, max_shard_size="5GB")
                model_to_log_mlflow = merged_model
            else:
                print(f"Saving PEFT adapter model to {final_model_output_dir}")
                best_peft_model.save_pretrained(final_model_output_dir, safe_serialization=True)
                model_to_log_mlflow = best_peft_model # Log the PEFT model (adapters)

            # Save tokenizer alongside the model
            tokenizer.save_pretrained(final_model_output_dir)
            print(f"Tokenizer saved to {final_model_output_dir}")

            # Log model to MLflow
            print("Attempting to log model to MLflow...")
            try:
                # Create a sample input for signature inference
                sample_input_text = "Human: Classify this job description: We need a great developer."
                signature = None
                input_example = pd.DataFrame([{"messages": [{"role": "user", "content": "Classify this job."}]}])
                # For PEFT models, output of generate is usually the full chat.
                # Let's construct an example assuming the model generates a category.
                output_example = pd.DataFrame([{"generated_text": "Assistant: Software Engineer"}])

                signature = infer_signature(input_example, output_example)


                mlflow.transformers.log_model(
                    transformers_model={ # This structure is robust for both PEFT and merged
                        "model": model_to_log_mlflow,
                        "tokenizer": tokenizer
                    },
                    artifact_path="fine_tuned_model_transformers", # Artifact path in MLflow
                    signature=signature
                )
                print("Model logged to MLflow using mlflow.transformers.log_model.")
            except Exception as e:
                print(f"Could not log model using mlflow.transformers.log_model: {e}")
                print("Logging model files as raw artifacts to MLflow...")
                mlflow.log_artifacts(final_model_output_dir, artifact_path="fine_tuned_model_files_raw")
                print("Logged model files as raw artifacts instead.")
    else:
        # Fallback if MLflow details are not provided (should not happen in pipeline)
        print("MLflow details not provided. Training and saving model locally (no MLflow logging).")
        trainer.train()
        best_peft_model = trainer.model # Assuming load_best_model_at_end=True
        final_model_output_dir = args.model_dir
        os.makedirs(final_model_output_dir, exist_ok=True)
        if args.merge_weights:
            # Simplified merge and save (same logic as above)
            temp_adapter_dir = "/tmp/temp_peft_adapter_for_merge"
            best_peft_model.save_pretrained(temp_adapter_dir)
            del model, trainer, best_peft_model; torch.cuda.empty_cache(); accelerator.free_memory()
            base_model_for_merge = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16 if args.bf16 else torch.float16, device_map="auto")
            merged_model = get_peft_model(base_model_for_merge, lora_config_obj)
            merged_model.load_adapter(temp_adapter_dir, adapter_name="default")
            merged_model = merged_model.merge_and_unload()
            merged_model.save_pretrained(final_model_output_dir, safe_serialization=True, max_shard_size="5GB")
        else:
            best_peft_model.save_pretrained(final_model_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_model_output_dir)

    print(f"Fine-tuning script completed. Model artifacts saved in {args.model_dir}")


if __name__ == "__main__":
    main()