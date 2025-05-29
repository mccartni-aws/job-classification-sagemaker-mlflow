import argparse
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel # If loading adapters separately
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from tqdm import tqdm # For progress bar

def extract_category_from_output(generated_text: str, prompt_template: str):
    """
    Extracts the category from the LLM's generated text.
    This needs to be robust. Assumes the category is the last part after the prompt.
    """
    # A simple way is to find where the prompt ends in the generated text
    # and take the rest. Or, if the model is well-behaved, it's just the output.
    # For "Assistant: <CATEGORY>"
    if "Assistant:" in generated_text:
         # Take text after the last "Assistant:"
         parts = generated_text.split("Assistant:")
         if len(parts) > 1:
             return parts[-1].strip().split('\n')[0].strip() # Get first line after last Assistant:
    # Fallback if the template isn't perfectly matched
    return generated_text.strip()


def evaluate_model(
    model_s3_path_or_mlflow_uri: str, # S3 path to merged model or MLflow model URI
    test_data_s3_path: str,
    poc_categories_s3_path: str, # S3 path to the poc_categories.json
    local_temp_model_dir: str = "/tmp/model_for_eval",
    local_temp_data_dir: str = "/tmp/data_for_eval",
    batch_size: int = 8, # Batch size for inference
    mlflow_arn: str = None,
    experiment_name: str = None,
    run_id: str = None # Parent run_id from pipeline
):
    os.makedirs(local_temp_model_dir, exist_ok=True)
    os.makedirs(local_temp_data_dir, exist_ok=True)

    # --- 1. Load Model and Tokenizer ---
    print(f"Loading model from: {model_s3_path_or_mlflow_uri}")
    if model_s3_path_or_mlflow_uri.startswith("s3://"):
        # Download from S3 if it's a path to saved model files
        # This part needs to be robust, e.g. aws s3 sync
        # For simplicity, assuming direct load if transformers supports s3 path or it's local after download
        model_load_path = model_s3_path_or_mlflow_uri
        # If it's a directory on S3, you'd need to download it first to local_temp_model_dir
        # e.g. using subprocess:
        # import subprocess
        # subprocess.run(["aws", "s3", "sync", model_s3_path_or_mlflow_uri, local_temp_model_dir], check=True)
        # model_load_path = local_temp_model_dir
        # For this example, assuming transformers can handle S3 path or it's already local
        tokenizer = AutoTokenizer.from_pretrained(model_load_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_load_path,
            torch_dtype=torch.bfloat16, # Or float16
            device_map="auto",
            trust_remote_code=True,
            cache_dir="/tmp/.cache_eval"
        )
    elif model_s3_path_or_mlflow_uri.startswith("runs:/"): # MLflow model URI
        print(f"Loading model from MLflow URI: {model_s3_path_or_mlflow_uri}")
        # Note: mlflow.transformers.load_model might expect a specific structure
        # or you might need to load the base model and apply PEFT adapters if logged that way
        try:
             loaded_model_data = mlflow.transformers.load_model(model_s3_path_or_mlflow_uri)
             model = loaded_model_data["model"]
             tokenizer = loaded_model_data["tokenizer"]
        except Exception as e:
            print(f"Standard MLflow load failed: {e}. Trying to load files from artifact path.")
            # Fallback: get artifact path and load manually
            client = mlflow.tracking.MlflowClient()
            run_id_mlflow = model_s3_path_or_mlflow_uri.split('/')[1] # runs:/<run_id>/...
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
    else: # Assuming local path
        tokenizer = AutoTokenizer.from_pretrained(model_s3_path_or_mlflow_uri)
        model = AutoModelForCausalLM.from_pretrained(
             model_s3_path_or_mlflow_uri,
             torch_dtype=torch.bfloat16,
             device_map="auto",
             trust_remote_code=True,
             cache_dir="/tmp/.cache_eval"
         )

    model.eval() # Set model to evaluation mode

    # --- 2. Load Test Data and Categories ---
    print(f"Loading test data from: {test_data_s3_path}")
    test_df = pd.read_json(test_data_s3_path, lines=True)

    # Load POC categories (used for confusion matrix labels)
    # Download categories file
    s3 = boto3.client('s3')
    bucket_name = poc_categories_s3_path.split('/')[2]
    key = '/'.join(poc_categories_s3_path.split('/')[3:])
    local_categories_file = os.path.join(local_temp_data_dir, "poc_categories.json")
    s3.download_file(bucket_name, key, local_categories_file)
    with open(local_categories_file, 'r') as f:
        poc_categories = json.load(f)
    print(f"Loaded {len(poc_categories)} POC categories.")


    true_labels = []
    predicted_labels = []
    raw_inputs = []

    # Use Hugging Face pipeline for easier batching if possible
    # Otherwise, manual batching and generation
    # For classification, the chat template is important
    tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE # Ensure it's set if not part of saved tokenizer

    print("Starting inference on test set...")
    for i in tqdm(range(0, len(test_df), batch_size)):
        batch_df = test_df.iloc[i:i+batch_size]
        prompts_batch = []
        for _, row in batch_df.iterrows():
            # Assuming 'messages' format from preprocess script
            # We need the user's message (the prompt for the LLM)
            user_message_content = row['messages'][0]['content'] # User is the first message
            # Format for generation - model expects the full chat history up to assistant's turn
            # For Llama 3, with add_generation_prompt=True, it appends "Assistant: "
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_message_content}],
                tokenize=False,
                add_generation_prompt=True # Critical for Llama-3 instruct
            )
            prompts_batch.append(prompt)
            true_labels.append(row['messages'][1]['content']) # Assistant is the second message (true label)
            raw_inputs.append(user_message_content)


        inputs = tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=30, # Max length for category name + some buffer
                pad_token_id=tokenizer.eos_token_id, # Important for generation
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False, # Deterministic for eval
                temperature=0.1
            )

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for idx, full_output_text in enumerate(decoded_outputs):
            # Extract the generated part after the prompt
            # The prompt itself will be part of full_output_text
            # We need to get only the newly generated part
            input_prompt_for_decode_stripping = prompts_batch[idx]
            # A common way to get only generated part:
            generated_part = full_output_text[len(input_prompt_for_decode_stripping):]
            pred_category = generated_part.strip().split('\n')[0].strip() # Take first line of generation
            predicted_labels.append(pred_category)

    # --- 3. Calculate Metrics ---
    # Filter out predictions not in known categories for metric calculation, or handle them
    # This is important as LLM might hallucinate.
    # For simplicity, we'll proceed, but in practice, you might map unknown to a special class or clean them.
    valid_predictions_indices = [i for i, label in enumerate(predicted_labels) if label in poc_categories]
    filtered_true_labels = [true_labels[i] for i in valid_predictions_indices]
    filtered_predicted_labels = [predicted_labels[i] for i in valid_predictions_indices]
    
    num_invalid_preds = len(predicted_labels) - len(filtered_predicted_labels)
    print(f"Number of predictions not in known categories: {num_invalid_preds} (out of {len(predicted_labels)})")


    if not filtered_predicted_labels: # No valid predictions
        print("No valid predictions found. Skipping metric calculation.")
        accuracy = 0.0
        report_dict = {}
        cm = None
    else:
        accuracy = accuracy_score(filtered_true_labels, filtered_predicted_labels)
        report_dict = classification_report(filtered_true_labels, filtered_predicted_labels, output_dict=True, labels=poc_categories, zero_division=0)
        cm = confusion_matrix(filtered_true_labels, filtered_predicted_labels, labels=poc_categories)

    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(filtered_true_labels, filtered_predicted_labels, labels=poc_categories, zero_division=0))

    # --- 4. Log to MLflow ---
    if mlflow_arn and experiment_name and run_id:
        mlflow.set_tracking_uri(mlflow_arn)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_id=run_id, nested=True) as run:
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("num_invalid_predictions", num_invalid_preds)
            if report_dict:
                for category, metrics in report_dict.items():
                    if isinstance(metrics, dict): # Per-class metrics
                        for metric_name, value in metrics.items():
                            mlflow.log_metric(f"{category}_{metric_name}".replace(" ", "_"), value)
                    else: # Overall metrics like macro avg, weighted avg
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

            # Log some example predictions
            example_preds_df = pd.DataFrame({
                "input_text": raw_inputs[:20], # Log first 20 raw inputs
                "true_label": true_labels[:20],
                "predicted_label": predicted_labels[:20]
            })
            example_preds_path = "/tmp/example_predictions.csv"
            example_preds_df.to_csv(example_preds_path, index=False)
            mlflow.log_artifact(example_preds_path, "evaluation_outputs")

    print("Evaluation complete.")
    return {"accuracy": accuracy, "classification_report": report_dict if report_dict else {}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_s3_path_or_mlflow_uri", type=str, required=True)
    parser.add_argument("--test_data_s3_path", type=str, required=True)
    parser.add_argument("--poc_categories_s3_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    # MLflow args
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