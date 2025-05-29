import sagemaker
from sagemaker.huggingface import HuggingFace
import time
import json
import boto3 # To get region for image URI

def launch_hf_training_job(
    train_file: str , # Expected filename within the S3 URI / channel
    validation_file: str, # Expected filename
    role: str,
    image_uri: str, # Pre-retrieved HuggingFace DLC image URI
    instance_type: str,
    instance_count: int,
    # Data related
    train_s3_uri: str,
    validation_s3_uri: str,
    # Model and training script params
    entry_point_script: str, # e.g., "finetune_entrypoint.py"
    source_directory: str,   # e.g., "steps/" or "steps/training_scripts/"
    model_id_hf: str,
    epochs_val: int,
    per_device_train_batch_size_val: int,
    learning_rate_val: float,
    lora_r_val: int,
    lora_alpha_val: int,
    lora_dropout_val: float,
    lora_target_modules_val: str,
    merge_weights_val: bool,
    hf_token_val: str,
    # MLflow params
    mlflow_tracking_arn: str,
    mlflow_experiment: str,
    pipeline_run_id: str, # This will be the MLflow parent run ID
    # Other SageMaker params
    base_job_name_prefix: str
):
    """
    Launches a SageMaker HuggingFace training job.
    """
    hyperparameters = {
        'model_id': model_id_hf,
        # train_data_dir and eval_data_dir are implicitly /opt/ml/input/data/training and /opt/ml/input/data/validation
        'train_file_name': train_file,
        'eval_file_name': validation_file,
        'epochs': epochs_val,
        'per_device_train_batch_size': per_device_train_batch_size_val,
        'learning_rate': learning_rate_val,
        'lora_r': lora_r_val,
        'lora_alpha': lora_alpha_val,
        'lora_dropout': lora_dropout_val,
        'lora_target_modules': lora_target_modules_val,
        'merge_weights': merge_weights_val, # Pass as bool
        'bf16': True, # Assuming bf16, can be parameterized
        'gradient_checkpointing': True, # Assuming True, can be parameterized
        'max_seq_length': 1024, # Can be parameterized
        'logging_steps': 10    # Can be parameterized
    }
    if hf_token_val and hf_token_val != "OPTIONAL_HF_TOKEN_PLACEHOLDER":
        hyperparameters['hf_token'] = hf_token_val
    if mlflow_tracking_arn:
        hyperparameters['mlflow_arn'] = mlflow_tracking_arn
    if mlflow_experiment:
        hyperparameters['experiment_name'] = mlflow_experiment
    if pipeline_run_id:
        hyperparameters['run_id'] = pipeline_run_id

    training_job_name = f'{base_job_name_prefix}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())}'

    huggingface_estimator = HuggingFace(
        entry_point=entry_point_script,
        source_dir=source_directory,
        instance_type=instance_type,
        instance_count=instance_count,
        role=role,
        transformers_version='4.36', # Match image
        pytorch_version='2.1',       # Match image
        py_version='py310',          # Match image
        image_uri=image_uri,
        hyperparameters=hyperparameters,
        metric_definitions=[ # Example, adjust to your script's logging
            {'Name': 'train:loss', 'Regex': "'loss': ([0-9\\.]+)"},
            {'Name': 'eval:loss', 'Regex': "'eval_loss': ([0-9\\.]+)"},
            {'Name': 'eval:accuracy', 'Regex': "'eval_accuracy': ([0-9\\.]+)"} # If your trainer logs this
        ],
        environment={"HUGGINGFACE_HUB_CACHE": "/tmp/.cache", "TRANSFORMERS_CACHE": "/tmp/.cache"},
        base_job_name=training_job_name,
        volume_size=200, # Adjust as needed
        keep_alive_period_in_seconds=300, # For spot training if used
        disable_profiler=True # Profiler can add overhead
    )

    # Define data channels
    # The keys 'training' and 'validation' will create directories:
    # /opt/ml/input/data/training and /opt/ml/input/data/validation inside the container.
    inputs = {
        'training': sagemaker.inputs.TrainingInput(
            s3_data=train_s3_uri,
            distribution='FullyReplicated',
            content_type='application/jsonlines', # Assuming JSONL
            s3_data_type='S3Prefix'
        ),
        'validation': sagemaker.inputs.TrainingInput(
            s3_data=validation_s3_uri,
            distribution='FullyReplicated',
            content_type='application/jsonlines',
            s3_data_type='S3Prefix'
        )
    }

    print(f"Starting SageMaker Training Job with name: {huggingface_estimator.base_job_name}")
    print(f"Hyperparameters: {json.dumps(hyperparameters, indent=2)}")
    print(f"Inputs: {inputs}")

    huggingface_estimator.fit(inputs, wait=True)

    training_job_description = huggingface_estimator.latest_training_job.describe()
    model_s3_artifacts = training_job_description['ModelArtifacts']['S3ModelArtifacts']
    
    print(f"Training job {huggingface_estimator.latest_training_job.job_name} completed.")
    print(f"Model artifacts saved to: {model_s3_artifacts}")

    return {
        "training_job_name": huggingface_estimator.latest_training_job.job_name,
        "s3_model_artifacts": model_s3_artifacts,
        "mlflow_run_id": pipeline_run_id # Pass this through for consistency
    }