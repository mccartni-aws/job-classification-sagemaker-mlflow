import argparse
import json
import os
import torch
import bitsandbytes as bnb # Make sure this is imported if used for bnb.nn.Linear4bit
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset # load_dataset can load local jsonl
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
import pandas as pd # For loading JSONL if needed, though datasets.load_dataset is better

# LLAMA_3_CHAT_TEMPLATE (same as before)
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] + eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] + eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
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
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def main_train_function(args):
    if args.hf_token:
        print(f"Logging into Hugging Face Hub with token: {args.hf_token[:10]}...")
        login(token=args.hf_token)

    accelerator = Accelerator()

    # --- 1. Load Tokenizer and Model ---
    print(f"Loading model and tokenizer for: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_auth_token=bool(args.hf_token))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE

    # --- 2. Load and Prepare Datasets ---
    # SageMaker maps input channels to /opt/ml/input/data/<channel_name>
    # The train_file and eval_file are expected to be within these channel directories.
    train_file_path = os.path.join(args.train_data_dir, args.train_file_name)
    eval_file_path = os.path.join(args.eval_data_dir, args.eval_file_name)

    print(f"Loading training data from: {train_file_path}")
    print(f"Loading evaluation data from: {eval_file_path}")

    try:
        train_dataset_raw = load_dataset("json", data_files=train_file_path, split="train")
        eval_dataset_raw = load_dataset("json", data_files=eval_file_path, split="train") # load_dataset loads 'train' split by default
    except Exception as e:
        print(f"Error loading datasets: {e}")
        print("Ensure train_file_name and eval_file_name are correct and present in their respective data directories.")
        raise

    def template_dataset_fn(examples):
        return {"text": tokenizer.apply_chat_template(examples["messages"], tokenize=False, add_generation_prompt=False)}

    with accelerator.main_process_first():
        tokenized_train_dataset = train_dataset_raw.map(
            template_dataset_fn, remove_columns=list(train_dataset_raw.features)
        ).map(
            lambda sample: tokenizer(sample["text"], truncation=True, max_length=args.max_seq_length),
            batched=True, remove_columns=["text"]
        )
        tokenized_eval_dataset = eval_dataset_raw.map(
            template_dataset_fn, remove_columns=list(eval_dataset_raw.features)
        ).map(
            lambda sample: tokenizer(sample["text"], truncation=True, max_length=args.max_seq_length),
            batched=True, remove_columns=["text"]
        )

    print(f"Processed train samples: {len(tokenized_train_dataset)}")
    print(f"Processed eval samples: {len(tokenized_eval_dataset)}")
    if len(tokenized_train_dataset) > 0: print("Sample train instance input_ids:", tokenized_train_dataset[0]['input_ids'][:20])


    # --- 3. Setup Model (QLoRA) ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map={'':torch.cuda.current_device()},
        trust_remote_code=True,
        use_auth_token=bool(args.hf_token),
        cache_dir="/tmp/.cache" # Use a writable cache dir in the container
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    modules = find_all_linear_names(model) if not args.lora_target_modules else args.lora_target_modules.split(',')
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

    # --- 4. Setup Trainer ---
    training_args = TrainingArguments(
        output_dir="/tmp/outputs", # Temp dir for trainer checkpoints etc.
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size, # Add if not present
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch", # Save best model based on eval loss
        load_best_model_at_end=True,
        save_total_limit=1,
        bf16=args.bf16,
        report_to="mlflow" if args.mlflow_arn else "none",
        run_name=f"{args.model_id.split('/')[-1]}-classification-peft" # MLflow run name for trainer
    )

    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False

    # --- 5. Train and Save with MLflow ---
    if args.mlflow_arn and args.experiment_name and args.run_id:
        mlflow.set_tracking_uri(args.mlflow_arn)
        mlflow.set_experiment(args.experiment_name)
        with mlflow.start_run(run_id=args.run_id, nested=True) as run: # Use parent run_id, log as nested
            mlflow.log_param("model_id", args.model_id)
            mlflow.log_params({
                "epochs": args.epochs, "learning_rate": args.learning_rate,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": args.lora_dropout,
                "target_modules": modules, "max_seq_length": args.max_seq_length
            })

            print("Starting training...")
            trainer.train()
            print("Training finished.")

            # Save best model from trainer to a temporary path before merging (if applicable)
            # The trainer.save_model() saves to training_args.output_dir / "checkpoint-best" or similar
            # If load_best_model_at_end=True, trainer.model is already the best model.
            
            model_to_save = trainer.model # This is the best PEFT model if load_best_model_at_end=True
            
            # SageMaker expects the final model in /opt/ml/model
            final_output_dir = "/opt/ml/model" 
            os.makedirs(final_output_dir, exist_ok=True)

            if args.merge_weights:
                print("Merging LoRA weights with base model...")
                # Need to reload the base model in a way that allows merging,
                # if the current 'model' is the PEFT-wrapped one in 4-bit.
                # This often requires reloading the base model in fp16/bf16.
                
                # Save PEFT adapter first
                temp_adapter_dir = "/tmp/peft_adapter"
                model_to_save.save_pretrained(temp_adapter_dir, safe_serialization=False) # safe_serialization=False for 4-bit
                
                del model, trainer # Clear memory
                torch.cuda.empty_cache()

                # Reload base model in higher precision for merging
                base_model_for_merge = AutoModelForCausalLM.from_pretrained(
                    args.model_id,
                    torch_dtype=torch.bfloat16 if args.bf16 else torch.float16, # Ensure correct dtype
                    device_map="auto", # device_map={'':torch.cuda.current_device()}
                    trust_remote_code=True,
                    use_auth_token=bool(args.hf_token),
                    cache_dir="/tmp/.cache"
                )
                
                # Load PEFT model from saved adapter and merge
                merged_model = AutoPeftModelForCausalLM.from_pretrained( # This implicitly loads base and adapter
                     temp_adapter_dir, # Path to the saved PEFT adapter
                     torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                     device_map="auto",
                     low_cpu_mem_usage=True,
                     trust_remote_code=True,
                     use_auth_token=bool(args.hf_token),
                     model_id_for_base_model=args.model_id # Explicitly provide base model id
                )
                merged_model = merged_model.merge_and_unload()
                
                print(f"Saving merged model to {final_output_dir}")
                merged_model.save_pretrained(final_output_dir, safe_serialization=True, max_shard_size="2GB")
                model_to_log_mlflow = merged_model
            else:
                print(f"Saving PEFT adapter model to {final_output_dir}")
                model_to_save.save_pretrained(final_output_dir, safe_serialization=True)
                model_to_log_mlflow = model_to_save # Log the PEFT model itself

            # Save tokenizer
            tokenizer.save_pretrained(final_output_dir)
            print(f"Tokenizer saved to {final_output_dir}")

            # Log model to MLflow
            print("Logging model to MLflow...")
            signature_params = {"max_new_tokens": 50, "temperature": 0.1, "do_sample": False}
            # Create a sample input for signature inference
            sample_input_text = "Classify the following job description. Job Description: We need a great developer."
            # To infer signature, model needs to be on device if not already
            try:
                # If model_to_log_mlflow is a PEFT model, it needs to be loaded with base model for generation.
                # For simplicity, if merged, it's straightforward. If not merged, logging might be more complex.
                # Assuming model_to_log_mlflow is ready for inference or is the merged model.
                model_for_sig = model_to_log_mlflow.to(accelerator.device) if hasattr(model_to_log_mlflow, 'to') else model_to_log_mlflow
                
                input_ids_for_sig = tokenizer(sample_input_text, return_tensors="pt").input_ids.to(model_for_sig.device)
                # Ensure generation params are compatible
                gen_kwargs_for_sig = {**signature_params, "pad_token_id": tokenizer.eos_token_id}

                # If model_to_log_mlflow is a PeftModel and not merged, direct generation might be an issue for signature.
                # The transformers.log_model handles PeftModel correctly by saving adapters and base_model_name.
                
                # Create a dummy pipeline for signature if direct generation is complex for PEFT non-merged
                # This is more robust for PEFT models that are not merged
                if not args.merge_weights: # If we are logging PEFT adapters
                     # For PEFT models, logging the model and tokenizer dictionary is standard
                     # The signature might be inferred based on a text-generation pipeline
                     signature_input_example = pd.DataFrame({"inputs": [sample_input_text]})
                     # Output can be tricky to infer without running a pipeline. Let's provide a simple one.
                     signature_output_example = pd.DataFrame({"generated_text": ["Sample Category"]})
                     signature = infer_signature(signature_input_example, signature_output_example, params=signature_params)

                else: # If merged model
                    generated_outputs = model_for_sig.generate(input_ids_for_sig, **gen_kwargs_for_sig)
                    decoded_prediction = tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
                    # Extract actual prediction part
                    prediction_only = decoded_prediction[len(sample_input_text):].strip()

                    signature = infer_signature(
                        sample_input_text, # Input example
                        prediction_only,   # Output example
                        params=signature_params
                    )

                mlflow.transformers.log_model(
                    transformers_model={
                        "model": model_to_log_mlflow, # This can be PeftModel or PreTrainedModel
                        "tokenizer": tokenizer
                    },
                    artifact_path="fine_tuned_classifier_model", # Relative path within MLflow run
                    signature=signature,
                    model_config=signature_params,
                    # If not merging, PEFT automatically handles saving adapter_config.json etc.
                    # and mlflow.transformers.load_model can reload it.
                )
                print("Model logged to MLflow.")
            except Exception as e:
                print(f"Could not log model to MLflow with transformers.log_model: {e}")
                print("Attempting to log model files as raw artifacts...")
                mlflow.log_artifacts(final_output_dir, artifact_path="fine_tuned_classifier_model_files")
                print("Logged model files as artifacts instead.")
    else:
        print("MLflow details not provided. Skipping MLflow logging. Training and saving model locally.")
        trainer.train()
        # Save model to /opt/ml/model without MLflow (similar logic as above)
        # ... (implementation for non-MLflow save) ...
        final_output_dir = "/opt/ml/model"
        os.makedirs(final_output_dir, exist_ok=True)
        if args.merge_weights:
            # ... (merge logic as above) ...
            merged_model.save_pretrained(final_output_dir, safe_serialization=True, max_shard_size="2GB")
        else:
            trainer.model.save_pretrained(final_output_dir, safe_serialization=True)
        tokenizer.save_pretrained(final_output_dir)

    print(f"Fine-tuning script complete. Model saved in {final_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Data args
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--train_data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/training"), help="Directory for training data (mounted by SageMaker).")
    parser.add_argument("--train_file_name", type=str, default="train_dataset.jsonl", help="Name of the training data file.")
    parser.add_argument("--eval_data_dir", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION", "/opt/ml/input/data/validation"), help="Directory for evaluation data (mounted by SageMaker).")
    parser.add_argument("--eval_file_name", type=str, default="validation_dataset.jsonl", help="Name of the evaluation data file.")
    parser.add_argument("--max_seq_length", type=int, default=1024)

    # Training args
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1) # Added
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True) # Store True/False
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--bf16", type=bool, default=True) # Store True/False

    # LoRA args
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj", help="Comma-separated list of LoRA target modules.")

    def str_to_bool(value):
        if isinstance(value, bool): return value
        if value.lower() in ('yes', 'true', 't', 'y', '1'): return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'): return False
        else: raise argparse.ArgumentTypeError('Boolean value expected.')

    # Other args
    parser.add_argument("--merge_weights", type=str_to_bool, default=True) # Store True/False
    parser.add_argument("--hf_token", type=str, default=None)

    # MLflow args
    parser.add_argument("--mlflow_arn", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None, help="Parent MLflow run ID from pipeline.")
    
    args = parser.parse_args()


    main_train_function(args)