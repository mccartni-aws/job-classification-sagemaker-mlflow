# steps/preprocess_job_descriptions.py

import argparse
import os
import json
import boto3
import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import mlflow
import re

# --- Configuration ---
SYSTEM_MESSAGE = """You are Llama, an AI assistant. Your knowledge spans a wide range of topics, allowing you to answer questions with honesty and truthfulness."""
DEFAULT_JOB_DESC_COLUMN = "job_description_text" # Expected column name for JD text
DEFAULT_CATEGORY_COLUMN = "category_label"     # Expected column name for category
DEFAULT_TEST_SPLIT_FRACTION = 0.15
DEFAULT_VALIDATION_FROM_TRAIN_FRACTION = 0.15

# --- Helper Functions ---

def clean_job_description_text(text):
    """Clean job description text by removing any translation markers"""
    if not isinstance(text, str):
        return str(text)
    
    # Remove translation error markers (safety measure)
    cleaned = re.sub(r'\[[A-Z]{2,}_(?:UNTRANSLATED|NO_TRANSLATE_CLIENT|FAILED_\w+|ERROR)\]', '', text)
    cleaned = re.sub(r'\[[A-Z_]+\]', '', cleaned)  # Remove any other bracketed markers
    cleaned = ' '.join(cleaned.split())  # Clean up whitespace
    return cleaned
    
def create_conversation_format(sample, job_desc_col, category_col):
    """
    Formats a raw data sample into messages list for fine-tuning.
    SIMPLIFIED: Only user and assistant messages, no system message.
    """
    jd_text = sample[job_desc_col]
    category = sample[category_col]

    # Clean the text (safety measure)
    cleaned_jd_text = clean_job_description_text(jd_text)

    # The prompt for the LLM
    user_prompt = f"Classify the following job description. Job Description: {cleaned_jd_text}"

    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": category}
        ]
    }

def load_data_as_pandas(raw_dataset_identifier: str):
    """
    Load data using pandas instead of HuggingFace datasets to avoid session issues.
    """
    print(f"Loading data using pandas from: {raw_dataset_identifier}")
    
    if raw_dataset_identifier.startswith("s3://"):
        # Parse S3 URI
        s3_path = raw_dataset_identifier.replace("s3://", "")
        bucket, key = s3_path.split("/", 1)
        
        # Download from S3 to local temp file
        s3_client = boto3.client("s3")
        local_temp_file = "/tmp/raw_data.jsonl"
        
        print(f"Downloading from S3: bucket={bucket}, key={key}")
        s3_client.download_file(bucket, key, local_temp_file)
        
        # Read the downloaded file
        if local_temp_file.endswith('.jsonl') or local_temp_file.endswith('.json'):
            df = pd.read_json(local_temp_file, lines=True)
        elif local_temp_file.endswith('.csv'):
            df = pd.read_csv(local_temp_file)
        else:
            # Try JSON lines first, then regular JSON
            try:
                df = pd.read_json(local_temp_file, lines=True)
            except:
                df = pd.read_json(local_temp_file)
        
        # Clean up temp file
        os.remove(local_temp_file)
        
    elif os.path.exists(raw_dataset_identifier):
        # Local file
        if raw_dataset_identifier.endswith('.jsonl'):
            df = pd.read_json(raw_dataset_identifier, lines=True)
        elif raw_dataset_identifier.endswith('.csv'):
            df = pd.read_csv(raw_dataset_identifier)
        else:
            df = pd.read_json(raw_dataset_identifier)
    else:
        # Try to load as HuggingFace dataset name (fallback)
        # This might still cause issues, but worth trying
        try:
            from datasets import load_dataset
            hf_dataset = load_dataset(raw_dataset_identifier)
            df = hf_dataset['train'].to_pandas()
        except Exception as e:
            raise ValueError(f"Could not load dataset from {raw_dataset_identifier}. Error: {e}")
    
    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    return df

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
    if mlflow_arn:
        mlflow.set_tracking_uri(mlflow_arn)
        print(f"MLflow tracking URI set to: {mlflow_arn}")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
        print(f"MLflow experiment set to: {experiment_name}")

    mlflow_run_id = None

    # --- Load data using pandas ---
    try:
        df = load_data_as_pandas(raw_dataset_identifier)
    except Exception as e:
        print(f"Error loading dataset '{raw_dataset_identifier}': {e}")
        raise

    # Check if required columns exist
    if job_desc_column not in df.columns:
        raise ValueError(f"Column '{job_desc_column}' not found. Available columns: {list(df.columns)}")
    if category_column not in df.columns:
        raise ValueError(f"Column '{category_column}' not found. Available columns: {list(df.columns)}")

    # --- Create train/validation/test splits using pandas ---
    print("Creating train/validation/test splits...")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_split_fraction, 
        random_state=42, 
        stratify=df[category_column] if len(df[category_column].unique()) > 1 else None
    )
    
    # Second split: separate validation from training
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=validation_from_train_fraction, 
        random_state=42,
        stratify=train_val_df[category_column] if len(train_val_df[category_column].unique()) > 1 else None
    )
    
    # Apply max_samples_per_split if specified
    if max_samples_per_split:
        train_df = train_df.head(max_samples_per_split)
        val_df = val_df.head(max_samples_per_split)
        test_df = test_df.head(max_samples_per_split)
    
    print(f"Split sizes - Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    # --- Convert pandas dataframes to HuggingFace datasets and format ---
    processed_splits = {}
    
    for split_name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        # Convert to HuggingFace Dataset
        hf_dataset = Dataset.from_pandas(split_df, preserve_index=False)
        
        # Apply conversation formatting
        formatted_dataset = hf_dataset.map(
            lambda x: create_conversation_format(x, job_desc_column, category_column),
            remove_columns=[col for col in hf_dataset.column_names if col not in ["messages"]]
        ) 
        
        # Filter for valid conversation structure
        formatted_dataset = formatted_dataset.filter(lambda x: len(x["messages"]) == 2)
        
        processed_splits[split_name] = formatted_dataset
        print(f"Processed '{split_name}' split with {len(formatted_dataset)} samples.")

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
    local_base_dir = "/tmp/processed_data_jd"
    os.makedirs(local_base_dir, exist_ok=True)

    for split_name, data_to_save in processed_splits.items():
        if not data_to_save:
            print(f"Split '{split_name}' is empty. Skipping save and upload.")
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

    # --- MLflow Logging ---
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

            # Log metrics and paths
            for split_name, path in output_s3_paths.items():
                if split_name != "categories":
                    mlflow.log_param(f"{split_name}_output_s3_path", path)
                    if split_name in processed_splits and processed_splits[split_name]:
                        mlflow.log_metric(f"{split_name}_sample_count", len(processed_splits[split_name]))
                    mlflow.log_text(path, artifact_file=f"output_paths/{split_name}_dataset_s3_path.txt")

            mlflow.log_metric("num_unique_categories", len(poc_categories_list))
            if "categories" in output_s3_paths:
                mlflow.log_param("categories_output_s3_path", output_s3_paths["categories"])
            
            # Log output paths and categories
            mlflow.log_dict(output_s3_paths, "output_s3_paths.json")
            if poc_categories_list:
                temp_cat_list_file = os.path.join(local_base_dir, "processed_categories_list.json")
                with open(temp_cat_list_file, 'w') as f_cat_list:
                    json.dump(poc_categories_list, f_cat_list)
                mlflow.log_artifact(temp_cat_list_file, artifact_path="metadata")
                os.remove(temp_cat_list_file)

    # Clean up
    if os.path.exists(local_base_dir):
        for f_name in os.listdir(local_base_dir):
            try:
                os.remove(os.path.join(local_base_dir, f_name))
            except OSError:
                pass
        os.rmdir(local_base_dir)

    # --- Prepare return dictionary ---
    final_return_paths = {
        "train_data_s3_path": output_s3_paths.get("train"),
        "validation_data_s3_path": output_s3_paths.get("validation"),
        "test_data_s3_path": output_s3_paths.get("test"),
        "categories_s3_path": output_s3_paths.get("categories"),
        "mlflow_run_id": mlflow_run_id
    }
    final_return_paths = {k: v for k, v in final_return_paths.items() if v is not None or k == "mlflow_run_id"}

    print(f"Preprocessing function finished. Returning: {final_return_paths}")
    return final_return_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess job description data for LLM fine-tuning.")
    parser.add_argument("--raw_dataset_identifier", type=str, required=True,
                        help="Identifier for the raw dataset (e.g., S3 URI to a JSONL file, or local path).")
    parser.add_argument("--s3_output_bucket", type=str, required=True,
                        help="S3 bucket to upload processed data.")
    parser.add_argument("--s3_output_prefix", type=str, required=True,
                        help="S3 prefix for the processed data within the bucket.")
    parser.add_argument("--job_desc_column", type=str, default=DEFAULT_JOB_DESC_COLUMN,
                        help=f"Column name for job description text. Default: {DEFAULT_JOB_DESC_COLUMN}")
    parser.add_argument("--category_column", type=str, default=DEFAULT_CATEGORY_COLUMN,
                        help=f"Column name for category label. Default: {DEFAULT_CATEGORY_COLUMN}")
    parser.add_argument("--test_split_fraction", type=float, default=DEFAULT_TEST_SPLIT_FRACTION,
                        help=f"Fraction of data to use for the test set. Default: {DEFAULT_TEST_SPLIT_FRACTION}")
    parser.add_argument("--validation_from_train_fraction", type=float, default=DEFAULT_VALIDATION_FROM_TRAIN_FRACTION,
                        help=f"Fraction of training data to use for validation set. Default: {DEFAULT_VALIDATION_FROM_TRAIN_FRACTION}")
    parser.add_argument("--max_samples_per_split", type=int, default=None,
                        help="Maximum number of samples to keep per split. Default: None")
    parser.add_argument("--mlflow_arn", type=str, default=os.environ.get("MLFLOW_TRACKING_URI"),
                        help="MLflow tracking server ARN.")
    parser.add_argument("--experiment_name", type=str, default="job-desc-classification-preprocess",
                        help="MLflow experiment name.")
    parser.add_argument("--run_name", type=str, default="preprocess-job-data-standalone",
                        help="MLflow run name.")

    args = parser.parse_args()

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
    print(f"Preprocessing complete. Output: {returned_info}")