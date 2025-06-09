# utils/mlflow_callbacks.py

from transformers import TrainerCallback
import mlflow
import torch

class MLflowProgressiveCallback(TrainerCallback):
    """Custom callback to log metrics progressively to MLflow during training"""
    
    def __init__(self, log_every_n_steps=5):
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics (controlled by logging_steps)"""
        if logs is not None and state.global_step % self.log_every_n_steps == 0:
            
            # Log training metrics with step information
            for key, value in logs.items():
                if isinstance(value, (int, float)) and not key.startswith('eval_'):
                    try:
                        mlflow.log_metric(
                            key=f"train_{key}" if not key.startswith('train_') else key,
                            value=value,
                            step=state.global_step
                        )
                    except Exception as e:
                        print(f"Failed to log {key}: {e}")
            
            # Log epoch progress
            if hasattr(state, 'epoch') and state.epoch is not None:
                try:
                    mlflow.log_metric("epoch", state.epoch, step=state.global_step)
                except Exception as e:
                    print(f"Failed to log epoch: {e}")
            
            # Log learning rate if available
            if 'learning_rate' in logs:
                try:
                    mlflow.log_metric("learning_rate", logs['learning_rate'], step=state.global_step)
                except Exception as e:
                    print(f"Failed to log learning_rate: {e}")
            
            # Log GPU memory usage if available
            if torch.cuda.is_available():
                try:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    mlflow.log_metric("gpu_memory_allocated_gb", memory_allocated, step=state.global_step)
                    mlflow.log_metric("gpu_memory_reserved_gb", memory_reserved, step=state.global_step)
                except Exception as e:
                    print(f"Failed to log GPU memory: {e}")
    
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        """Called after evaluation"""
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)) and key.startswith('eval_'):
                    try:
                        mlflow.log_metric(key, value, step=state.global_step)
                    except Exception as e:
                        print(f"Failed to log eval metric {key}: {e}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training"""
        print(f"MLflow Progressive Callback initialized - logging every {self.log_every_n_steps} steps")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training"""
        print("Training completed - MLflow logging finished")

def create_mlflow_callback(log_every_n_steps=5):
    """Factory function to create MLflow callback"""
    return MLflowProgressiveCallback(log_every_n_steps=log_every_n_steps)