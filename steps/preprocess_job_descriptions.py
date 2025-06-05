# steps/preprocess_job_descriptions.py

import argparse
import os
import json
import boto3
import pandas as pd
from datasets import load_dataset, DatasetDict
# from sklearn.model_selection import train_test_split as sk_train_test_split # Not used, can remove
import mlflow

# --- Configuration ---
SYSTEM_MESSAGE = """You are Llama, an AI assistant. Your knowledge spans a wide range of topics, allowing you to answer questions with honesty and truthfulness."""
DEFAULT_JOB_DESC_COLUMN = "job_description_text" # Expected column name for JD text
DEFAULT_CATEGORY_COLUMN = "category_label"     # Expected column name for category
DEFAULT_TEST_SPLIT_FRACTION = 0.15
DEFAULT_VALIDATION_FROM_TRAIN_FRACTION = 0.15

# --- Helper Functions ---
def create_conversation_format(sample, job_desc_col, category_col):
    """
    Formats a raw data sample into the initial messages list for fine-tuning.
    (System message will be added later by 'add_system_message_to_conversation')
    """
    jd_text = sample[job_desc_col]
    category = sample[category_col]

    # The prompt for the LLM
    user_prompt = f"Classify the following job description. Job Description: {jd_text}"

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": category}
        ]
    }

def add_system_message_to_conversation(sample):
    """
    Prepends the system message to the conversation if not already present.
    """
    if sample["messages"] and sample["messages"][0]["role"] == "system":
        return sample # System message already exists
    else:
        sample["messages"] = [{"role": "system", "content": SYSTEM_MESSAGE}] + sample["messages"]
        return sample

def preprocess_job_data(
    raw_dataset_identifier: str,
    s3_output_bucket: str,
    s3_output_prefix: str,
    job_desc_column: str = DEFAULT_JOB_DESC_COLUMN,
    category_column: str = DEFAULT_CATEGORY_COLUMN,
    test_split_fraction: float = DEFAULT_TEST_SPLIT_FRACTION,
    validation_from_train_fraction: float = DEFAULT_VALIDATION_FROM_TRAIN_FRACTION,
    max_samples_per_split: int = None,
    mlflow_arn: str = None,
    experiment_name: str = None,
    run_name: str = None,
):
    """
    Loads raw data, formats it for Llama 3 fine-tuning, splits, uploads to S3,
    and logs artifacts and metadata to MLflow.
    """
    print(f"Starting preprocessing for dataset: {raw_dataset_identifier}")
    print(f"Output S3 Location: s3://{s3_output_bucket}/{s3_output_prefix}")

    # --- MLflow Setup ---
    # Set tracking URI and experiment name if provided
    # This setup is done before starting a run, similar to preprocess_llama3.py
    if mlflow_arn:
        mlflow.set_tracking_uri(mlflow_arn)
        print(f"MLflow tracking URI set to: {mlflow_arn}")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set to: {experiment_name}")

    mlflow_run_id = None # Initialize mlflow_run_id

    print(f"Loading raw dataset from: {raw_dataset_identifier}")
    try:
        if raw_dataset_identifier.startswith("s3://"):
            print(f"Attempting to load from S3 path: {raw_dataset_identifier} as JSONL.")
            # Assuming a single JSONL file. For more complex structures, adjust data_files.
            raw_hf_dataset = load_dataset("json", data_files={"train": raw_dataset_identifier}, split="train")
            raw_hf_dataset = DatasetDict({"train": raw_hf_dataset}) # Ensure DatasetDict structure
        else:
            raw_hf_dataset = load_dataset(raw_dataset_identifier)
    except Exception as e:
        print(f"Error loading dataset '{raw_dataset_identifier}': {e}")
        raise

    # --- Ensure standard splits (train, validation, test) ---
    processed_splits = {}
    if "train" not in raw_hf_dataset:
        raise ValueError("Raw dataset must contain at least a 'train' split.")

    temp_train_data = raw_hf_dataset["train"]

    # Create test split if not present
    if "test" in raw_hf_dataset:
        processed_splits["test"] = raw_hf_dataset["test"]
    else:
        print(f"No 'test' split found. Creating one from 'train' with fraction: {test_split_fraction}")
        split_output = temp_train_data.train_test_split(test_size=test_split_fraction, shuffle=True, seed=42)
        temp_train_data = split_output["train"] # Update temp_train_data
        processed_splits["test"] = split_output["test"]

    # Create validation split from (remaining) train data if not present
    if "validation" in raw_hf_dataset:
        processed_splits["validation"] = raw_hf_dataset["validation"]
        processed_splits["train"] = temp_train_data # Use the (potentially reduced by test split) train data
    else:
        print(f"No 'validation' split found. Creating one from remaining 'train' data with fraction: {validation_from_train_fraction}")
        # Ensure validation_from_train_fraction is applied to the current temp_train_data size
        if len(temp_train_data) * validation_from_train_fraction < 1:
             raise ValueError(f"Train data size ({len(temp_train_data)}) is too small to create a validation split with fraction {validation_from_train_fraction}.")
        split_output = temp_train_data.train_test_split(test_size=validation_from_train_fraction, shuffle=True, seed=42)
        processed_splits["train"] = split_output["train"]
        processed_splits["validation"] = split_output["test"] # 'test' from this split becomes our validation

    # --- Apply formatting and system message ---
    print("Formatting data and adding system messages...")
    for split_name in ["train", "validation", "test"]:
        if split_name not in processed_splits:
            print(f"Warning: Expected split '{split_name}' not found after processing. Skipping.")
            continue
        
        hf_split_data = processed_splits[split_name]
        
        # Check if required columns exist before mapping
        if job_desc_column not in hf_split_data.column_names or \
           category_column not in hf_split_data.column_names:
            raise ValueError(
                f"Required columns '{job_desc_column}' or '{category_column}' not found in '{split_name}' split. "
                f"Available columns: {hf_split_data.column_names}"
            )

        # 1. Create user/assistant messages
        formatted_split = hf_split_data.map(
            create_conversation_format,
            fn_kwargs={"job_desc_col": job_desc_column, "category_col": category_column},
            remove_columns=[col for col in hf_split_data.column_names if col not in ["messages"]]
        )
        # 2. Add system message
        formatted_split = formatted_split.map(add_system_message_to_conversation)
        # 3. Filter for valid conversation structure
        formatted_split = formatted_split.filter(lambda x: len(x["messages"][1:]) % 2 == 0)

        # 4. (Optional) Sub-sample
        if max_samples_per_split and len(formatted_split) > max_samples_per_split:
            print(f"Sub-sampling '{split_name}' split to {max_samples_per_split} samples.")
            formatted_split = formatted_split.select(range(max_samples_per_split))

        processed_splits[split_name] = formatted_split
        print(f"Processed '{split_name}' split with {len(formatted_split)} samples.")

    # --- Collect unique categories from training data ---
    all_categories = set()
    if "train" in processed_splits:
        for example in processed_splits["train"]:
            for message in example["messages"]:
                if message["role"] == "assistant":
                    all_categories.add(message["content"])
    poc_categories_list = sorted(list(all_categories))
    print(f"Found {len(poc_categories_list)} unique categories in the training data: {poc_categories_list}")

    # --- Save to local JSONL and upload to S3 ---
    s3_client = boto3.client("s3")
    output_s3_paths = {}
    local_base_dir = "/tmp/processed_data_jd" # Use a unique temp dir name
    os.makedirs(local_base_dir, exist_ok=True)

    for split_name, data_to_save in processed_splits.items():
        if not data_to_save: # Skip if a split is empty after processing
            print(f"Split '{split_name}' is empty or was not processed. Skipping save and upload.")
            continue
        local_file_path = os.path.join(local_base_dir, f"{split_name}_dataset.jsonl")
        s3_key = f"{s3_output_prefix}/{split_name}/{split_name}_dataset.jsonl"

        data_to_save.to_json(local_file_path, orient="records", lines=True, force_ascii=False)
        s3_client.upload_file(local_file_path, s3_output_bucket, s3_key)
        output_s3_paths[split_name] = f"s3://{s3_output_bucket}/{s3_key}"
        print(f"Uploaded {split_name} data to {output_s3_paths[split_name]}")
        os.remove(local_file_path)

    # Save categories list
    if poc_categories_list:
        local_categories_file = os.path.join(local_base_dir, "poc_categories.json")
        s3_categories_key = f"{s3_output_prefix}/poc_categories.json"
        with open(local_categories_file, 'w') as f:
            json.dump(poc_categories_list, f, indent=2)
        s3_client.upload_file(local_categories_file, s3_output_bucket, s3_categories_key)
        output_s3_paths["categories"] = f"s3://{s3_output_bucket}/{s3_categories_key}"
        print(f"Uploaded categories list to {output_s3_paths['categories']}")
        os.remove(local_categories_file)
    else:
        print("No categories found to save.")

    # --- MLflow Logging ---
    # Start a run if mlflow_arn, experiment_name, AND run_name are provided
    if mlflow_arn and experiment_name and run_name:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow_run_id = run.info.run_id
            print(f"MLflow Run ID: {mlflow_run_id}")

            # Log parameters
            mlflow.log_param("raw_dataset_identifier", raw_dataset_identifier)
            mlflow.log_param("job_desc_column", job_desc_column)
            mlflow.log_param("category_column", category_column)
            mlflow.log_param("s3_output_prefix", s3_output_prefix)
            mlflow.log_param("test_split_fraction", test_split_fraction)
            mlflow.log_param("validation_from_train_fraction", validation_from_train_fraction)
            if max_samples_per_split:
                mlflow.log_param("max_samples_per_split", max_samples_per_split)

            # Log metrics (sample counts) and S3 paths for datasets
            for split_name, path in output_s3_paths.items():
                if split_name != "categories": # Categories path is logged separately
                    mlflow.log_param(f"{split_name}_output_s3_path", path)
                    if split_name in processed_splits and processed_splits[split_name]:
                         mlflow.log_metric(f"{split_name}_sample_count", len(processed_splits[split_name]))
                    # Log S3 path as a text artifact for easier access from MLflow UI
                    mlflow.log_text(path, artifact_file=f"output_paths/{split_name}_dataset_s3_path.txt")

            mlflow.log_metric("num_unique_categories", len(poc_categories_list))
            if "categories" in output_s3_paths:
                mlflow.log_param("categories_output_s3_path", output_s3_paths["categories"])
                mlflow.log_artifact(local_categories_file, artifact_path="metadata") # Or log the s3 path as text
            
            # Log all output paths as a dictionary artifact
            mlflow.log_dict(output_s3_paths, "output_s3_paths.json")
            # Log processed categories list as an artifact
            if poc_categories_list:
                 # Create a temporary local file for poc_categories_list to log as artifact
                temp_cat_list_file = os.path.join(local_base_dir, "processed_categories_list.json")
                with open(temp_cat_list_file, 'w') as f_cat_list:
                    json.dump(poc_categories_list, f_cat_list)
                mlflow.log_artifact(temp_cat_list_file, artifact_path="metadata")
                os.remove(temp_cat_list_file)

    elif not (mlflow_arn and experiment_name):
        print("MLflow tracking URI or experiment name not provided. Skipping MLflow run creation and logging.")
    elif not run_name:
        print("MLflow run_name not provided. Skipping MLflow run creation and logging.")

    # Clean up local temporary directory
    if os.path.exists(local_base_dir):
        for f_name in os.listdir(local_base_dir): # Ensure it's empty before rmdir if files were not removed
            try:
                os.remove(os.path.join(local_base_dir, f_name))
            except OSError:
                pass # ignore if file already removed
        os.rmdir(local_base_dir)

    # --- Prepare return dictionary ---
    # Ensure consistent keys for pipeline steps if needed, e.g., 'training_input_path'
    # For now, using descriptive keys based on splits.
    final_return_paths = {
        "train_data_s3_path": output_s3_paths.get("train"),
        "validation_data_s3_path": output_s3_paths.get("validation"),
        "test_data_s3_path": output_s3_paths.get("test"),
        "categories_s3_path": output_s3_paths.get("categories"),
        "mlflow_run_id": mlflow_run_id
    }
    # Filter out None paths if splits were not created/saved
    final_return_paths = {k: v for k, v in final_return_paths.items() if v is not None or k == "mlflow_run_id"}


    print(f"Preprocessing function finished. Returning: {final_return_paths}")
    return final_return_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess job description data for LLM fine-tuning.")
    parser.add_argument("--raw_dataset_identifier", type=str, required=True,
                        help="Identifier for the raw dataset (e.g., Hugging Face Hub ID, S3 URI to a JSONL file, or local path).")
    parser.add_argument("--s3_output_bucket", type=str, required=True,
                        help="S3 bucket to upload processed data.")
    parser.add_argument("--s3_output_prefix", type=str, required=True,
                        help="S3 prefix for the processed data within the bucket (e.g., 'processed_data/job_descriptions').")
    parser.add_argument("--job_desc_column", type=str, default=DEFAULT_JOB_DESC_COLUMN,
                        help=f"Column name for job description text. Default: {DEFAULT_JOB_DESC_COLUMN}")
    parser.add_argument("--category_column", type=str, default=DEFAULT_CATEGORY_COLUMN,
                        help=f"Column name for category label. Default: {DEFAULT_CATEGORY_COLUMN}")
    parser.add_argument("--test_split_fraction", type=float, default=DEFAULT_TEST_SPLIT_FRACTION,
                        help=f"Fraction of data to use for the test set if not present. Default: {DEFAULT_TEST_SPLIT_FRACTION}")
    parser.add_argument("--validation_from_train_fraction", type=float, default=DEFAULT_VALIDATION_FROM_TRAIN_FRACTION,
                        help=f"Fraction of training data to use for validation set if not present. Default: {DEFAULT_VALIDATION_FROM_TRAIN_FRACTION}")
    parser.add_argument("--max_samples_per_split", type=int, default=None,
                        help="Maximum number of samples to keep per split (for quick testing). Default: None (no limit).")
    parser.add_argument("--mlflow_arn", type=str, default=os.environ.get("MLFLOW_TRACKING_URI"),
                        help="MLflow tracking server ARN. Defaults to MLFLOW_TRACKING_URI env variable.")
    parser.add_argument("--experiment_name", type=str, default="job-desc-classification-preprocess",
                        help="MLflow experiment name. Default: 'job-desc-classification-preprocess'")
    parser.add_argument("--run_name", type=str, default="preprocess-job-data-standalone",
                        help="MLflow run name. Default: 'preprocess-job-data-standalone'")

    args = parser.parse_args()

    print(f"Starting preprocessing script with CLI args: {vars(args)}")

    returned_info = preprocess_job_data(
        raw_dataset_identifier=args.raw_dataset_identifier,
        s3_output_bucket=args.s3_output_bucket,
        s3_output_prefix=args.s3_output_prefix,
        job_desc_column=args.job_desc_column,
        category_column=args.category_column,
        test_split_fraction=args.test_split_fraction,
        validation_from_train_fraction=args.validation_from_train_fraction,
        max_samples_per_split=args.max_samples_per_split,
        mlflow_arn=args.mlflow_arn,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
    print(f"Preprocessing script execution complete. Output info: {returned_info}")