# steps/preprocess_job_descriptions.py

import argparse
import os
import json
import boto3
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split as sk_train_test_split # To avoid conflict
import mlflow

# --- Configuration ---
SYSTEM_MESSAGE = """You are Llama, an AI assistant. Your knowledge spans a wide range of topics, allowing you to answer questions with honesty and truthfulness."""
DEFAULT_JOB_DESC_COLUMN = "job_description_text" # Expected column name for JD text
DEFAULT_CATEGORY_COLUMN = "category_label"     # Expected column name for category

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

def preprocess_data(
    raw_dataset_identifier: str,
    s3_output_bucket: str,
    s3_output_prefix: str,
    job_desc_column: str = DEFAULT_JOB_DESC_COLUMN,
    category_column: str = DEFAULT_CATEGORY_COLUMN,
    test_split_fraction: float = 0.15, # Fraction of total for test
    validation_from_train_fraction: float = 0.15, # Fraction of (original_train) for validation
    max_samples_per_split: int = None, # Optional: limit samples for quick tests
    mlflow_arn: str = None,
    experiment_name: str = None,
    run_name: str = None,
):
    """
    Loads raw data, formats it for Llama 3 fine-tuning, splits, and uploads to S3.
    Logs artifacts and metadata to MLflow.
    """

    if mlflow_arn and experiment_name:
        mlflow.set_tracking_uri(mlflow_arn)
        mlflow.set_experiment(experiment_name)
    else:
        print("MLflow ARN or experiment name not provided. Skipping MLflow setup.")

    print(f"Loading raw dataset from: {raw_dataset_identifier}")
    # Load dataset - this can be a Hugging Face Hub ID, or path to local files/S3 (if s3fs installed and format is right)
    # For S3, load_dataset might need specific data_files arguments, e.g., load_dataset("json", data_files="s3://...")
    try:
        if raw_dataset_identifier.startswith("s3://"):
             # Assuming a single JSONL file for simplicity here.
             # For multiple files or directories, data_files arg needs more complex handling.
            print(f"Attempting to load from S3 path: {raw_dataset_identifier} as JSONL.")
            raw_hf_dataset = load_dataset("json", data_files={"train": raw_dataset_identifier}, split="train")
            # Convert to DatasetDict for consistent handling later
            raw_hf_dataset = DatasetDict({"train": raw_hf_dataset})
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

    if "test" in raw_hf_dataset:
        processed_splits["test"] = raw_hf_dataset["test"]
    else:
        print(f"No 'test' split found. Creating one from 'train' with fraction: {test_split_fraction}")
        split_output = temp_train_data.train_test_split(test_size=test_split_fraction, shuffle=True, seed=42)
        temp_train_data = split_output["train"]
        processed_splits["test"] = split_output["test"]

    if "validation" in raw_hf_dataset:
        processed_splits["validation"] = raw_hf_dataset["validation"]
        processed_splits["train"] = temp_train_data # Use the (potentially reduced by test split) train data
    else:
        print(f"No 'validation' split found. Creating one from remaining 'train' data with fraction: {validation_from_train_fraction}")
        split_output = temp_train_data.train_test_split(test_size=validation_from_train_fraction, shuffle=True, seed=42)
        processed_splits["train"] = split_output["train"]
        processed_splits["validation"] = split_output["test"] # 'test' from this split becomes our validation

    # --- Apply formatting and system message ---
    print("Formatting data and adding system messages...")
    for split_name, hf_split_data in processed_splits.items():
        # 1. Create user/assistant messages
        formatted_split = hf_split_data.map(
            create_conversation_format,
            fn_kwargs={"job_desc_col": job_desc_column, "category_col": category_column},
            remove_columns=[col for col in hf_split_data.column_names if col not in ["messages"]] # Keep only 'messages'
        )
        # 2. Add system message
        formatted_split = formatted_split.map(add_system_message_to_conversation)

        # 3. Filter for valid conversation structure (even number of user/assistant turns after system)
        formatted_split = formatted_split.filter(lambda x: len(x["messages"][1:]) % 2 == 0)

        # 4. (Optional) Sub-sample for quick testing
        if max_samples_per_split and len(formatted_split) > max_samples_per_split:
            print(f"Sub-sampling '{split_name}' split to {max_samples_per_split} samples.")
            formatted_split = formatted_split.select(range(max_samples_per_split))

        processed_splits[split_name] = formatted_split
        print(f"Processed '{split_name}' split with {len(formatted_split)} samples.")

    # --- Collect unique categories ---
    all_categories = set()
    # Iterating through the 'assistant' message content in the processed training data
    for example in processed_splits["train"]:
        for message in example["messages"]:
            if message["role"] == "assistant":
                all_categories.add(message["content"])
    poc_categories_list = sorted(list(all_categories))
    print(f"Found {len(poc_categories_list)} unique categories in the training data.")

    # --- Save to local JSONL and upload to S3 ---
    s3_client = boto3.client("s3")
    output_s3_paths = {}
    local_base_dir = "/tmp/processed_data"
    os.makedirs(local_base_dir, exist_ok=True)

    for split_name, data_to_save in processed_splits.items():
        local_file_path = os.path.join(local_base_dir, f"{split_name}_dataset.jsonl")
        s3_key = f"{s3_output_prefix}/{split_name}/{split_name}_dataset.jsonl"

        data_to_save.to_json(local_file_path, orient="records", lines=True, force_ascii=False)
        s3_client.upload_file(local_file_path, s3_output_bucket, s3_key)
        output_s3_paths[split_name] = f"s3://{s3_output_bucket}/{s3_key}"
        print(f"Uploaded {split_name} data to {output_s3_paths[split_name]}")
        os.remove(local_file_path)

    # Save categories list
    local_categories_file = os.path.join(local_base_dir, "poc_categories.json")
    s3_categories_key = f"{s3_output_prefix}/poc_categories.json"
    with open(local_categories_file, 'w') as f:
        json.dump(poc_categories_list, f, indent=2)
    s3_client.upload_file(local_categories_file, s3_output_bucket, s3_categories_key)
    output_s3_paths["categories"] = f"s3://{s3_output_bucket}/{s3_categories_key}"
    print(f"Uploaded categories list to {output_s3_paths['categories']}")
    os.remove(local_categories_file)


    # --- MLflow Logging ---
    mlflow_run_id = None
    if mlflow_arn and experiment_name and run_name:
        with mlflow.start_run(run_name=run_name) as run:
            mlflow_run_id = run.info.run_id
            print(f"MLflow Run ID: {mlflow_run_id}")

            mlflow.log_param("raw_dataset_identifier", raw_dataset_identifier)
            mlflow.log_param("job_desc_column", job_desc_column)
            mlflow.log_param("category_column", category_column)
            mlflow.log_param("s3_output_prefix", s3_output_prefix)

            for split_name, path in output_s3_paths.items():
                if split_name != "categories": # Log datasets, not the category list as mlflow.data
                    mlflow.log_param(f"{split_name}_output_s3_path", path)
                    mlflow.log_metric(f"{split_name}_sample_count", len(processed_splits[split_name]))
                    
                    # Log as MLflow data asset
                    # Need to load from S3 into pandas to use mlflow.data.from_pandas
                    # df_split = pd.read_json(path, orient="records", lines=True)
                    # mlflow_data_asset = mlflow.data.from_pandas(df_split, source=path)
                    # mlflow.log_input(mlflow_data_asset, context=split_name)
                    # Simpler: log path as artifact or param, full data logging can be large
                    mlflow.log_text(path, artifact_file=f"{split_name}_dataset_s3_path.txt")


            mlflow.log_metric("num_unique_categories", len(poc_categories_list))
            mlflow.log_dict({"s3_output_paths": output_s3_paths}, "output_s3_paths.json")
            mlflow.log_list(poc_categories_list, "processed_categories.json")

    # Ensure the returned dictionary matches what the pipeline expects
    # The SageMaker pipeline notebook expects 'train', 'validation', 'test' keys for data paths.
    # 'categories' path is also useful for evaluation.
    final_return_paths = {
        "train": output_s3_paths.get("train"),
        "validation": output_s3_paths.get("validation"),
        "test": output_s3_paths.get("test"),
        "categories_s3_path": output_s3_paths.get("categories"), # Added for clarity
        "mlflow_run_id": mlflow_run_id
    }
    return final_return_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dataset_identifier", type=str, required=True,
                        help="Identifier for the raw dataset (e.g., Hugging Face Hub ID, S3 URI to a JSONL file, or local path).")
    parser.add_argument("--s3_output_bucket", type=str, required=True,
                        help="S3 bucket to upload processed data.")
    parser.add_argument("--s3_output_prefix", type=str, required=True,
                        help="S3 prefix for the processed data within the bucket.")
    parser.add_argument("--job_desc_column", type=str, default=DEFAULT_JOB_DESC_COLUMN,
                        help="Column name for job description text in the raw dataset.")
    parser.add_argument("--category_column", type=str, default=DEFAULT_CATEGORY_COLUMN,
                        help="Column name for category label in the raw dataset.")
    parser.add_argument("--test_split_fraction", type=float, default=0.15)
    parser.add_argument("--validation_from_train_fraction", type=float, default=0.15)
    parser.add_argument("--max_samples_per_split", type=int, default=None,
                        help="Maximum number of samples to keep per split (for quick testing).")
    parser.add_argument("--mlflow_arn", type=str, default=os.environ.get("MLFLOW_TRACKING_URI"))
    parser.add_argument("--experiment_name", type=str, default="job-desc-classification-preprocess")
    parser.add_argument("--run_name", type=str, default="preprocess-raw-data")

    args = parser.parse_args()

    print(f"Starting preprocessing with args: {args}")

    returned_paths = preprocess_data(
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
    print(f"Preprocessing complete. Output S3 paths and MLflow run ID: {returned_paths}")