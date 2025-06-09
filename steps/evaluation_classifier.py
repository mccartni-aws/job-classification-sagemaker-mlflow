import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logs
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['DISABLE_MLFLOW_INTEGRATION'] = 'true'  # Disable MLflow's TF integration

# This prevents transformers from importing TensorFlow unnecessarily
os.environ['USE_TF'] = 'NO'
os.environ['USE_TORCH'] = 'YES'

# Import transformers with torch backend only
import transformers
transformers.utils.logging.set_verbosity_error()

import torch

import argparse
import json
import torch
import boto3  # ADD: Missing import
from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from tqdm import tqdm

# ADD: Missing chat template
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)

def download_and_extract_model_from_s3(s3_path: str, local_dir: str):
    """Download and extract model from S3 (handles both tar.gz and directory formats)"""
    import tarfile
    
    # Clean the S3 path (remove trailing slashes)
    s3_path = s3_path.rstrip('/')
    
    # Ensure local directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    s3 = boto3.client('s3')
    bucket_name = s3_path.replace('s3://', '').split('/')[0]
    prefix = '/'.join(s3_path.replace('s3://', '').split('/')[1:])
    
    # Ensure prefix ends with / for proper directory listing
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    
    print(f"Checking S3 path: s3://{bucket_name}/{prefix}")
    
    # List objects to see what's available
    paginator = s3.get_paginator('list_objects_v2')
    objects = []
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                objects.append(obj['Key'])
    
    if not objects:
        raise ValueError(f"No objects found at S3 path: {s3_path}")
    
    print(f"Found {len(objects)} objects:")
    for obj in objects[:5]:
        print(f"  {obj}")
    
    # Check if there's a model.tar.gz file
    tar_gz_files = [obj for obj in objects if obj.endswith('model.tar.gz')]
    
    if tar_gz_files:
        # Download and extract the tar.gz file
        tar_gz_key = tar_gz_files[0]
        local_tar_path = os.path.join(local_dir, 'model.tar.gz')
        
        print(f"Downloading compressed model: {tar_gz_key}")
        s3.download_file(bucket_name, tar_gz_key, local_tar_path)
        
        print(f"Extracting model to: {local_dir}")
        with tarfile.open(local_tar_path, 'r:gz') as tar:
            tar.extractall(local_dir)
        
        # Remove the tar file to save space
        os.remove(local_tar_path)
        
        # List extracted files
        extracted_files = []
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), local_dir)
                extracted_files.append(rel_path)
        
        print(f"Extracted {len(extracted_files)} files:")
        for file in extracted_files[:10]:
            print(f"  {file}")
        
        return local_dir
        
    else:
        # Download individual files (original logic)
        print("No tar.gz found, downloading individual files...")
        downloaded_files = 0
        
        for obj_key in objects:
            if obj_key.endswith('/'):
                continue
            
            relative_path = obj_key[len(prefix):] if prefix else obj_key
            if not relative_path:
                continue
            
            local_file_path = os.path.join(local_dir, relative_path)
            parent_dir = os.path.dirname(local_file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            
            print(f"Downloading {obj_key} to {local_file_path}")
            s3.download_file(bucket_name, obj_key, local_file_path)
            downloaded_files += 1
        
        print(f"Downloaded {downloaded_files} files")
        return local_dir

def evaluate_model(
    model_s3_path_or_mlflow_uri: str,
    test_data_s3_path: str,
    poc_categories_s3_path: str,
    local_temp_model_dir: str = "/tmp/model_for_eval",
    local_temp_data_dir: str = "/tmp/data_for_eval",
    batch_size: int = 4,  # CHANGED: Reduced from 8 for memory
    mlflow_arn: str = None,
    experiment_name: str = None,
    run_id: str = None
):
    os.makedirs(local_temp_model_dir, exist_ok=True)
    os.makedirs(local_temp_data_dir, exist_ok=True)

    # --- 1. Load Model and Tokenizer ---
    print(f"Loading model from: {model_s3_path_or_mlflow_uri}")
    if model_s3_path_or_mlflow_uri.startswith("s3://"):
        try:
            # For SageMaker training outputs, we need to download and extract
            print("Downloading and extracting model from S3...")
            model_local_path = download_and_extract_model_from_s3(model_s3_path_or_mlflow_uri, local_temp_model_dir)
            
            # Check what files we have after extraction
            print("Files in extracted model directory:")
            for root, dirs, files in os.walk(model_local_path):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), model_local_path)
                    print(f"  {rel_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_local_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_local_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="/tmp/.cache_eval"
            )
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            raise
        
    elif model_s3_path_or_mlflow_uri.startswith("runs:/"):
        print(f"Loading model from MLflow URI: {model_s3_path_or_mlflow_uri}")
        try:
            loaded_model_data = mlflow.transformers.load_model(model_s3_path_or_mlflow_uri)
            model = loaded_model_data["model"]
            tokenizer = loaded_model_data["tokenizer"]
        except Exception as e:
            print(f"Standard MLflow load failed: {e}. Trying to load files from artifact path.")
            client = mlflow.tracking.MlflowClient()
            run_id_mlflow = model_s3_path_or_mlflow_uri.split('/')[1]
            artifact_path_mlflow = '/'.join(model_s3_path_or_mlflow_uri.split('/')[2:])
            local_download_path = client.download_artifacts(run_id_mlflow, artifact_path_mlflow, dst_path=local_temp_model_dir)
            print(f"Downloaded MLflow model artifacts to: {local_download_path}")
            
            tokenizer = AutoTokenizer.from_pretrained(local_download_path)
            model = AutoModelForCausalLM.from_pretrained(
                local_download_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="/tmp/.cache_eval"
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_s3_path_or_mlflow_uri)
        model = AutoModelForCausalLM.from_pretrained(
            model_s3_path_or_mlflow_uri,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="/tmp/.cache_eval"
        )

    model.eval()

    # --- 2. Load Test Data and Categories ---
    print(f"Loading test data from: {test_data_s3_path}")
    
    # Download test data from S3
    s3 = boto3.client('s3')
    bucket_name = test_data_s3_path.split('/')[2]
    key = '/'.join(test_data_s3_path.split('/')[3:])
    local_test_file = os.path.join(local_temp_data_dir, "test_data.jsonl")
    s3.download_file(bucket_name, key, local_test_file)
    
    test_df = pd.read_json(local_test_file, lines=True)

    # Load POC categories
    bucket_name_cat = poc_categories_s3_path.split('/')[2]
    key_cat = '/'.join(poc_categories_s3_path.split('/')[3:])
    local_categories_file = os.path.join(local_temp_data_dir, "poc_categories.json")
    s3.download_file(bucket_name_cat, key_cat, local_categories_file)
    
    with open(local_categories_file, 'r') as f:
        poc_categories = json.load(f)
    print(f"Loaded {len(poc_categories)} POC categories.")

    true_labels = []
    predicted_labels = []
    raw_inputs = []

    # Set chat template
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Starting inference on test set...")
    for i in tqdm(range(0, len(test_df), batch_size)):
        batch_df = test_df.iloc[i:i+batch_size]
        prompts_batch = []
        
        for _, row in batch_df.iterrows():
            # Extract user message and true label
            user_message_content = row['messages'][0]['content']
            true_label = row['messages'][1]['content']
            
            # Format for generation
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message_content}],
                tokenize=False,
                add_generation_prompt=True
            )
            
            prompts_batch.append(prompt)
            true_labels.append(true_label)
            raw_inputs.append(user_message_content)

        # Tokenize batch
        inputs = tokenizer(
            prompts_batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512  # CHANGED: Reduced from 1024 for memory
        ).to(model.device)

        # Generate predictions
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # CHANGED: Reduced from 30
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                temperature=0.1
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Extract predictions
        for idx, full_output_text in enumerate(decoded_outputs):
            input_prompt_for_decode_stripping = prompts_batch[idx]
            generated_part = full_output_text[len(input_prompt_for_decode_stripping):].strip()
            pred_category = generated_part.split('\n')[0].strip()
            predicted_labels.append(pred_category)

    # --- 3. Calculate Metrics ---
    # FIXED: Handle case where categories might not match exactly
    valid_predictions_indices = []
    for i, pred_label in enumerate(predicted_labels):
        if pred_label in poc_categories:
            valid_predictions_indices.append(i)
        else:
            # Try to find close matches (case insensitive, strip spaces)
            pred_clean = pred_label.lower().strip()
            for cat in poc_categories:
                if cat.lower().strip() == pred_clean:
                    predicted_labels[i] = cat  # Normalize to correct category
                    valid_predictions_indices.append(i)
                    break
    
    filtered_true_labels = [true_labels[i] for i in valid_predictions_indices]
    filtered_predicted_labels = [predicted_labels[i] for i in valid_predictions_indices]
    
    num_invalid_preds = len(predicted_labels) - len(filtered_predicted_labels)
    print(f"Number of predictions not in known categories: {num_invalid_preds} (out of {len(predicted_labels)})")

    if not filtered_predicted_labels:
        print("No valid predictions found. Skipping metric calculation.")
        accuracy = 0.0
        report_dict = {}
        cm = None
    else:
        accuracy = accuracy_score(filtered_true_labels, filtered_predicted_labels)
        report_dict = classification_report(
            filtered_true_labels, 
            filtered_predicted_labels, 
            output_dict=True, 
            labels=poc_categories, 
            zero_division=0
        )
        cm = confusion_matrix(
            filtered_true_labels, 
            filtered_predicted_labels, 
            labels=poc_categories
        )

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(
        filtered_true_labels, 
        filtered_predicted_labels, 
        labels=poc_categories, 
        zero_division=0
    ))

    # --- 4. Log to MLflow ---
    if mlflow_arn and experiment_name:
        mlflow.set_tracking_uri(mlflow_arn)
        mlflow.set_experiment(experiment_name)
        
        if run_id:
            with mlflow.start_run(run_id=run_id, nested=True) as run:
                log_metrics_to_mlflow(accuracy, num_invalid_preds, report_dict, cm, poc_categories, raw_inputs, true_labels, predicted_labels)
        else:
            with mlflow.start_run() as run:
                log_metrics_to_mlflow(accuracy, num_invalid_preds, report_dict, cm, poc_categories, raw_inputs, true_labels, predicted_labels)

    print("Evaluation complete.")
    return {
        "accuracy": accuracy, 
        "classification_report": report_dict if report_dict else {},
        "num_total_predictions": len(predicted_labels),
        "num_valid_predictions": len(filtered_predicted_labels)
    }

def log_metrics_to_mlflow(accuracy, num_invalid_preds, report_dict, cm, poc_categories, raw_inputs, true_labels, predicted_labels):
    """Helper function to log metrics to MLflow"""
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("num_invalid_predictions", num_invalid_preds)
    
    if report_dict:
        for category, metrics in report_dict.items():
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    mlflow.log_metric(f"{category}_{metric_name}".replace(" ", "_"), value)
            else:
                mlflow.log_metric(f"{category}".replace(" ", "_"), metrics)

    # Log confusion matrix
    if cm is not None:
        plt.figure(figsize=(max(10, len(poc_categories)//2), max(8, len(poc_categories)//2)))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=poc_categories, yticklabels=poc_categories, cmap="Blues")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        cm_path = "/tmp/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path, "evaluation_plots")
        plt.close()

    # Log example predictions
    example_preds_df = pd.DataFrame({
        "input_text": raw_inputs[:20],
        "true_label": true_labels[:20],
        "predicted_label": predicted_labels[:20]
    })
    example_preds_path = "/tmp/example_predictions.csv"
    example_preds_df.to_csv(example_preds_path, index=False)
    mlflow.log_artifact(example_preds_path, "evaluation_outputs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_s3_path_or_mlflow_uri", type=str, required=True)
    parser.add_argument("--test_data_s3_path", type=str, required=True)
    parser.add_argument("--poc_categories_s3_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mlflow_arn", type=str, default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment_name", type=str, default="job-desc-classification")
    parser.add_argument("--run_id", type=str, help="MLflow parent run ID from pipeline")

    args = parser.parse_args()

    results = evaluate_model(
        model_s3_path_or_mlflow_uri=args.model_s3_path_or_mlflow_uri,
        test_data_s3_path=args.test_data_s3_path,
        poc_categories_s3_path=args.poc_categories_s3_path,
        batch_size=args.batch_size,
        mlflow_arn=args.mlflow_arn,
        experiment_name=args.experiment_name,
        run_id=args.run_id
    )
    print(f"Evaluation results: {results}")