# LLM Fine-Tuning for Job Description Classification using SageMaker Pipelines & MLflow

## 🚀 Overview

This project demonstrates how to fine-tune a Large Language Model (LLM), specifically Llama 3, for a multi-class job description classification task. It leverages Amazon SageMaker Pipelines for orchestrating the MLOps workflow (preprocessing, fine-tuning, evaluation) and MLflow for experiment tracking and model management.

A key feature is the generation of synthetic, multilingual job description data using Amazon Translate for the Proof of Concept (POC).

## 🎯 Use Case

The target use case is for a CV/Job platform that needs to:
1. Classify job descriptions into a large, standardized set of occupation categories (e.g., ~2000 categories).
2. Handle job descriptions in multiple languages (e.g., English, Spanish, French).
3. Process documents in batch for cost-effective classification.

This repository provides a POC focusing on a subset of categories and demonstrating the end-to-end process: synthetic data generation, preprocessing, fine-tuning, and evaluation.

## 🛠️ Approach

- **Model:** Llama 3 8B (leveraging QLoRA for efficient fine-tuning).
- **Task Formulation:** The classification task is framed as an instruction-following/chat completion task. The LLM is prompted to identify the correct category for a given job description.
  - Example Prompt: `"Classify the following job description. Job Description: [JOB_DESCRIPTION_TEXT]"`
  - Expected Output: `"[CATEGORY_NAME]"`
- **Data Generation:**
  - A Python script (`scripts/python/generate_and_upload_raw_data.py`) generates synthetic job descriptions based on templates defined in `data/jd_templates.json`.
  - Amazon Translate is used to translate English templates into other specified languages (e.g., Spanish, French) to create a multilingual raw dataset.
  - This raw dataset (a JSONL file) is uploaded to S3.
- **Workflow Orchestration:** Amazon SageMaker Pipelines are used to define and execute the multi-step machine learning workflow using the S3 raw data as input.
- **Experiment Tracking:** MLflow is integrated to log parameters, metrics, and model artifacts for each experiment run, facilitating comparison and reproducibility.

## 📁 Repository Structure

```
.
├── data/
│   └── jd_templates.json                           # JSON file with job description templates
├── notebooks/
│   └── job_classification_llm_sagemaker_mlflow.ipynb # Main SageMaker Pipeline orchestration notebook
├── scripts/
│   ├── python/
│   │   └── generate_and_upload_raw_data.py         # Python script for synthetic data generation & S3 upload
│   └── bash/
│       ├── run_data_generation.sh                  # Bash script to run the Python data generator
│       └── setup_venv.sh                           # Bash script to set up local Python venv for data generation
├── steps/
│   ├── __init__.py                                 # Makes 'steps' a package
│   ├── preprocess_job_descriptions.py              # Script for data preprocessing (consumes raw S3 data)
│   ├── finetune_llama3_classifier.py               # Script for fine-tuning Llama 3
│   └── evaluation_classifier.py                    # Script for evaluating the fine-tuned model
├── README.md                                       # This file
└── (Optional: requirements.txt for SageMaker steps, Dockerfile for custom images)
```

## ✅ Prerequisites

### For Local Data Generation:
1. **Python 3.8+**: Installed locally.
2. **AWS CLI**: Configured with credentials that have permissions for:
   - `s3:PutObject` (to upload generated data to your SageMaker default S3 bucket)
   - `translate:TranslateText` (for Amazon Translate)
3. **Required Python Packages**: `boto3`, `sagemaker` (for default bucket access). The `scripts/bash/setup_venv.sh` script helps install these in a virtual environment.

### For SageMaker Pipeline Execution:
1. **AWS Account**: An active AWS account.
2. **IAM Role for SageMaker**: An IAM Role with necessary permissions:
   - `AmazonSageMakerFullAccess`
   - `AmazonS3FullAccess` (or more restricted access to specific buckets used by the pipeline)
   - Permissions to interact with your MLflow Tracking Server.
3. **SageMaker Environment**:
   - A SageMaker Studio domain or a SageMaker Notebook Instance.
   - The SageMaker Python SDK and other required libraries installed in the notebook environment (see notebook for pip installs).
4. **MLflow Tracking Server**:
   - An MLflow Tracking Server configured (e.g., SageMaker-managed MLflow). You'll need its ARN.
5. **Hugging Face Hub Account & Token**:
   - Required if using gated models like Llama 3.
   - An HF access token (`HUGGING_FACE_HUB_TOKEN`) with read access to the model.
6. **Model Access**: Ensure you have accepted the terms for `meta-llama/Meta-Llama-3-8B` (or your chosen model) on Hugging Face Hub.

## ⚙️ Setup Instructions

### 1. Local Environment Setup (for Data Generation)

This step is for running the synthetic data generation script locally.

```bash
# Navigate to the root of the cloned repository
cd /path/to/job-classification-sagemaker-mlflow

# Make the setup script executable
chmod +x scripts/bash/setup_venv.sh

# Run the setup script to create a Python virtual environment and install dependencies
./scripts/bash/setup_venv.sh

# Activate the virtual environment (if not already activated by the script)
source llm_data_gen_venv/bin/activate 
# On Windows, it would be: llm_data_gen_venv\Scripts\activate
```

Ensure your AWS CLI is configured with appropriate credentials and default region.

### 2. Generate and Upload Raw Synthetic Data

Use the provided bash script to run the Python data generator. This script will create multilingual synthetic JDs and upload them to your SageMaker default S3 bucket.

```bash
# Ensure the virtual environment from Step 1 is activated
# source llm_data_gen_venv/bin/activate

# Make the data generation runner script executable
chmod +x scripts/bash/run_data_generation.sh

# Run the data generation script (example with default parameters)
./scripts/bash/run_data_generation.sh

# Example with custom parameters:
# ./scripts/bash/run_data_generation.sh \
#   --num-jds 15 \
#   --languages "en,de" \
#   --s3-prefix "my_company/raw_jd_data_german" \
#   --aws-region "eu-central-1" \
#   --templates-file "./data/jd_templates.json"
```

The script will output the S3 URI of the generated `raw_jds_translated.jsonl` file. Copy this S3 URI.

### 3. Configure SageMaker Pipeline Notebook

1. **Clone the Repository** (if not already done) or upload files to SageMaker.

2. **Open the Notebook**: Navigate to `notebooks/job_classification_llm_sagemaker_mlflow.ipynb` in your SageMaker environment.

3. **Set MLflow Tracking Server ARN**:
   - Locate the `mlflow_tracking_server_arn` variable in cell [3].
   - Replace the placeholder with the ARN of your MLflow Tracking Server.

4. **Set Raw Data S3 URI**:
   - Locate `param_raw_data_s3_uri` in cell [4] (Pipeline Parameters).
   - Update its `default_value` with the S3 URI of the `raw_jds_translated.jsonl` file you obtained from Step 2. Alternatively, you can pass this URI when starting the pipeline execution.

5. **Set Hugging Face Token**:
   - The pipeline parameter `param_hf_token` (cell [4]) can be used.
   - Ensure its `default_value` is set to your HF token if you want to use it directly (less secure for shared notebooks), or pass it during `pipeline.start()`. The placeholder is `OPTIONAL_HF_TOKEN_PLACEHOLDER`.
   - Alternatively, ensure the `HF_TOKEN` environment variable is available to the SageMaker fine-tuning job environment.

## 🚀 Running the SageMaker Pipeline

1. **Select Kernel**: In the SageMaker notebook, choose an appropriate kernel (e.g., "Python 3 (Data Science)").

2. **Execute Cells**: Run the notebook cells sequentially.
   - The notebook installs dependencies, sets up configurations, defines pipeline parameters, and defines the pipeline steps.
   - It then upserts (creates or updates) the pipeline definition in SageMaker and starts an execution.

3. **Override Parameters (Optional)**: When calling `pipeline.start()`, you can override any defined pipeline parameters:

```python
execution = pipeline.start(
    parameters={
        "RawDatasetS3URI": "s3://your-actual-bucket/your-actual-raw-data.jsonl",
        "FineTuneEpochs": 2,
        "LoraR": 16
        # ... any other parameters
    }
)
```

4. **Monitor Execution**: Track pipeline progress in the SageMaker console under "Pipelines".

## 🧩 Pipeline Steps

The SageMaker Pipeline consists of:

### Preprocessing (`steps/preprocess_job_descriptions.py`):
- Loads the raw multilingual job description data from the S3 URI specified by `RawDatasetS3URI`.
- Formats data for Llama 3 (system message, user/assistant turns).
- Splits data into train, validation, and test sets.
- Uploads formatted datasets and a list of unique categories (`poc_categories.json`) to S3.
- Logs dataset info to MLflow.

### Fine-tuning (`steps/finetune_llama3_classifier.py`):
- Loads preprocessed train/validation data from S3.
- Loads the base Llama 3 model.
- Applies QLoRA for parameter-efficient fine-tuning.
- Saves the fine-tuned model artifacts to S3 (via `/opt/ml/model`).
- Logs training parameters, metrics, and the model to MLflow.

### Evaluation (`steps/evaluation_classifier.py`):
- Loads the fine-tuned model (typically via its MLflow URI).
- Loads preprocessed test data and `poc_categories.json` from S3.
- Performs inference and calculates metrics (accuracy, classification report, confusion matrix).
- Logs evaluation metrics and visualizations to MLflow.

## 📊 Experiment Tracking with MLflow

- All runs, parameters, metrics, and artifacts are logged to your configured MLflow Tracking Server.
- The SageMaker Pipeline Execution ID is used as the parent `run_id` in MLflow, with each step logging as a nested run or under this parent.
- Track parameters, training/validation metrics, evaluation metrics, and artifacts like S3 paths, model files, and confusion matrices.

## ✨ Key Features & Considerations

- **Multilingual Synthetic Data**: Uses Amazon Translate for generating translated job descriptions.
- **Externalized Templates**: Job description templates are managed in `data/jd_templates.json`.
- **QLoRA Efficiency**: Reduces resources for fine-tuning.
- **SageMaker Function Steps (@step)**: Flexible pipeline definition.
- **Custom Docker Images/DLCs**: The pipeline uses a SageMaker Hugging Face PyTorch DLC for GPU steps. Ensure it includes all necessary libraries.

## 🔮 Future Work & Improvements

- **Advanced Data Generation**: Use LLMs for more diverse synthetic JD generation directly in target languages.
- **Robust Error Handling**: Add more comprehensive error handling in all scripts.
- **Full Category Set**: Scale data and fine-tuning for a larger number of categories.
- **Deployment**: Add pipeline steps for SageMaker endpoint deployment or Batch Transform.
- **Cost Optimization**: Fine-tune instance types and configurations.
- **Security**: Retrieve sensitive tokens like `HF_TOKEN` from AWS Secrets Manager.

This README provides a guide to understanding, setting up, and running the LLM fine-tuning pipeline for job description classification.