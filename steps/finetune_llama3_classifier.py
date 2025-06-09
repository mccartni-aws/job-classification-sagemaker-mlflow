import sagemaker
from sagemaker.huggingface import HuggingFace
import time
import json
import boto3
import mlflow
import os

def finetune_classifier_hf(
    preprocess_step_output: dict,
    train_config: dict,
    lora_config: dict,
    role: str,
    mlflow_arn: str,
    experiment_name: str,
    # run_name from pipeline (ExecutionVariables.PIPELINE_EXECUTION_ID) is expected
    # to be in preprocess_step_output['mlflow_run_id']
    *args # For SageMaker step decorator if any extra args are passed
):
    """
    Sets up and launches a SageMaker HuggingFace training job for fine-tuning a classifier.
    Logs parameters to an existing MLflow run.

    Args:
        preprocess_step_output (dict): Output from the preprocessing step.
            Expected keys: "train_data_s3_path", "validation_data_s3_path", "mlflow_run_id".
        train_config (dict): Configuration for the training job.
            Expected keys: "model_id_hf", "instance_type", "instance_count", "epochs",
                           "per_device_train_batch_size", "learning_rate",
                           "entry_point_script", "source_directory", "base_job_name_prefix",
                           "image_uri", "hf_token" (optional),
                           "transformers_version", "pytorch_version", "py_version".
        lora_config (dict): Configuration for LoRA.
            Expected keys: "lora_r", "lora_alpha", "lora_dropout",
                           "lora_target_modules", "merge_weights".
        role (str): IAM role for SageMaker.
        mlflow_arn (str): MLflow tracking server ARN.
        experiment_name (str): MLflow experiment name.
    """

    print("--- Starting Fine-tuning Step ---")
    # --- Parse JSON string configurations if they are strings ---
    # This is crucial if train_config or lora_config are ParameterString in the pipeline
    if isinstance(train_config, str):
        print("train_config is a string, attempting to parse as JSON.")
        train_config = json.loads(train_config)
    if isinstance(lora_config, str):
        print("lora_config is a string, attempting to parse as JSON.")
        lora_config = json.loads(lora_config)
        
    print(f"Preprocessing output: {json.dumps(preprocess_step_output, indent=2)}")
    print(f"Train config: {json.dumps(train_config, indent=2)}")
    print(f"LoRA config: {json.dumps(lora_config, indent=2)}")

    # --- MLflow Setup ---
    mlflow.set_tracking_uri(mlflow_arn)
    mlflow.set_experiment(experiment_name)

    # Use the run_id from the preprocessing step to continue logging in the same MLflow run
    parent_mlflow_run_id = preprocess_step_output.get("mlflow_run_id")
    if not parent_mlflow_run_id:
        raise ValueError("mlflow_run_id not found in preprocess_step_output. Cannot continue MLflow run.")

    with mlflow.start_run(run_id=parent_mlflow_run_id) as run:
        print(f"Continuing MLflow run with ID: {run.info.run_id}")
        mlflow.set_tag("Stage", "FineTuning")

        # --- Extract and Log Parameters ---
        mlflow.log_params({"ft_train_" + k: v for k, v in train_config.items()})
        mlflow.log_params({"ft_lora_" + k: v for k, v in lora_config.items()})
        mlflow.log_param("ft_sagemaker_role", role)

        # --- Prepare Data Inputs ---
        train_s3_full_path = preprocess_step_output.get("train_data_s3_path")
        validation_s3_full_path = preprocess_step_output.get("validation_data_s3_path")

        if not train_s3_full_path or not validation_s3_full_path:
            raise ValueError("Training or validation S3 path not found in preprocess_step_output.")

        # Extract S3 prefix and filename for training data
        train_s3_uri_prefix = os.path.dirname(train_s3_full_path)
        train_file_name = os.path.basename(train_s3_full_path)

        # Extract S3 prefix and filename for validation data
        validation_s3_uri_prefix = os.path.dirname(validation_s3_full_path)
        validation_file_name = os.path.basename(validation_s3_full_path)

        print(f"Training data S3 prefix: {train_s3_uri_prefix}, file: {train_file_name}")
        print(f"Validation data S3 prefix: {validation_s3_uri_prefix}, file: {validation_file_name}")

        # --- Prepare Hyperparameters for the Training Script ---
        hyperparameters = {
            'model_id': train_config["model_id"],
            'train_file_name': train_file_name,
            'eval_file_name': validation_file_name,
            'epochs': train_config["epoch"],
            'per_device_train_batch_size': train_config["per_device_train_batch_size"],
            'lr': train_config["learning_rate"],
            'lora_r': lora_config["lora_r"],
            'lora_alpha': lora_config["lora_alpha"],
            'lora_dropout': lora_config["lora_dropout"],
            'merge_weights': bool(lora_config["merge_weights"]),
            'bf16': train_config.get('bf16', True),
            'gradient_checkpointing': train_config.get('gradient_checkpointing', True),
            'max_seq_length': train_config.get('max_seq_length', 512),
            
            # UPDATED: Progressive logging parameters
            'logging_steps': train_config.get('logging_steps', 5),
            'eval_steps': train_config.get('eval_steps', 10),
            'gradient_accumulation_steps': train_config.get('gradient_accumulation_steps', 4),
            'limit_train_samples': train_config.get('limit_train_samples', None),
            'limit_eval_samples': train_config.get('limit_eval_samples', None),
            
            # MLflow details
            'mlflow_arn': mlflow_arn,
            'experiment_name': experiment_name,
            'run_id': parent_mlflow_run_id
        }
        if train_config.get("hf_token") and train_config["hf_token"] != "OPTIONAL_HF_TOKEN_PLACEHOLDER":
            hyperparameters['hf_token'] = train_config["hf_token"]

        # --- SageMaker Estimator Setup ---
        training_job_name = f'huggingface-qlora-{train_config["epoch"]}-{lora_config["lora_r"]}-{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}'

        huggingface_estimator = HuggingFace(
            entry_point=train_config["entry_point_script"],
            source_dir=train_config["source_directory"],
            instance_type=train_config["finetune_instance_type"],
            instance_count=train_config["finetune_num_instances"],
            role=role,
            image_uri = f"763104351884.dkr.ecr.{boto3.session.Session().region_name}.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04",
            py_version=train_config.get("py_version", 'py310'),
            hyperparameters=hyperparameters,
            metric_definitions=train_config.get("metric_definitions", [ # Example, adjust to your script's logging
                {'Name': 'train:loss', 'Regex': "'loss': ([0-9\\.]+)"},
                {'Name': 'eval:loss', 'Regex': "'eval_loss': ([0-9\\.]+)"},
                {'Name': 'eval:accuracy', 'Regex': "'eval_accuracy': ([0-9\\.]+)"}
            ]),
            environment={"HUGGINGFACE_HUB_CACHE": "/tmp/.cache", "TRANSFORMERS_CACHE": "/tmp/.cache"},
            base_job_name=training_job_name, # This will be the unique job name
            volume_size=train_config.get("volume_size", 200),
            keep_alive_period_in_seconds=train_config.get("keep_alive_period_in_seconds", 300),
            disable_profiler=train_config.get("disable_profiler", True)
        )

        # Define data channels for SageMaker Training Job
        # The keys 'training' and 'validation' create directories:
        # /opt/ml/input/data/training and /opt/ml/input/data/validation in the container.
        # The training script will then use `train_file_name` and `eval_file_name`
        # to access the specific files within these directories.
        inputs = {
            'training': sagemaker.inputs.TrainingInput(
                s3_data=train_s3_uri_prefix, # S3 prefix for training data
                distribution='FullyReplicated',
                content_type='application/jsonlines', # Assuming JSONL based on preprocess_job_descriptions.py
                s3_data_type='S3Prefix'
            ),
            'validation': sagemaker.inputs.TrainingInput(
                s3_data=validation_s3_uri_prefix, # S3 prefix for validation data
                distribution='FullyReplicated',
                content_type='application/jsonlines',
                s3_data_type='S3Prefix'
            )
        }

        print(f"--- Launching SageMaker Training Job ---")
        print(f"Estimator base_job_name: {huggingface_estimator.base_job_name}")
        print(f"Hyperparameters for script: {json.dumps(hyperparameters, indent=2)}")
        print(f"Input channels: {inputs}")

        huggingface_estimator.fit(inputs, wait=True)

        # --- Post-Training ---
        actual_training_job_name = huggingface_estimator.latest_training_job.job_name
        training_job_description = huggingface_estimator.latest_training_job.describe()
        model_s3_artifacts = training_job_description['ModelArtifacts']['S3ModelArtifacts']

        print(f"Training job {actual_training_job_name} completed.")
        print(f"Model artifacts saved to: {model_s3_artifacts}")

        mlflow.log_param("ft_sagemaker_actual_job_name", actual_training_job_name)
        mlflow.log_param("ft_sagemaker_model_artifacts_s3", model_s3_artifacts)
        # Log the model artifacts as an MLflow artifact (link) if desired
        # mlflow.log_artifact(model_s3_artifacts, artifact_path="sagemaker_model_output") # This logs the s3 path as a file

    print("--- Fine-tuning Step Completed ---")

    return {
        "training_job_name": actual_training_job_name,
        "s3_model_artifacts": model_s3_artifacts,
        "mlflow_run_id": parent_mlflow_run_id # Pass through the run_id for subsequent steps
    }