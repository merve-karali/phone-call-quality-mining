# Databricks notebook source
# MAGIC %md
# MAGIC # Register Parakeet ASR Model with MLflow 3
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Creates a custom PyFunc wrapper for the Parakeet ASR model
# MAGIC 2. Logs the model to Unity Catalog using MLflow 3
# MAGIC 3. Registers the model for serving endpoint deployment

# COMMAND ----------

# MAGIC %pip install pydub mutagen numpy>=1.24
# MAGIC %pip install -U nemo_toolkit["asr"] mlflow torch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import base64
import io
import logging
import os
import tempfile
from typing import List

import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration (Inline)

# COMMAND ----------

# Configuration - inline values
CATALOG = "<your-catalog>"
SCHEMA = "<your-schema>"

# MLflow 3 - Set Unity Catalog as the model registry
mlflow.set_registry_uri("databricks-uc")

# Model configuration
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.parakeet_asr_model"

print("=" * 80)
print("Model Registration Configuration")
print("=" * 80)
print(f"Source Model: {MODEL_NAME}")
print(f"Unity Catalog Model: {UC_MODEL_NAME}")
print(f"MLflow Version: {mlflow.__version__}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Define Custom PyFunc Model Wrapper

# COMMAND ----------

class ParakeetASRModel(mlflow.pyfunc.PythonModel):
    """
    Custom PyFunc wrapper for NVIDIA Parakeet ASR model.
    """
    
    def load_context(self, context):
        """Load the Parakeet model when the serving endpoint starts."""
        import logging
        import torch
        from nemo.collections.asr.models import EncDecCTCModelBPE
        
        logging.getLogger("nemo_logger").setLevel(logging.ERROR)
        logging.getLogger("nemo.collections").setLevel(logging.ERROR)
        logging.getLogger("nemo.core").setLevel(logging.ERROR)
        
        print("Loading Parakeet ASR model...")
        self.model = EncDecCTCModelBPE.from_pretrained(
            model_name="nvidia/parakeet-tdt-0.6b-v3"
        )
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Model loaded on CPU")
        
        self.model.eval()
        print("Model ready for inference")
    
    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        import base64
        import os
        import tempfile
        import torch
        
        results = []
        
        for idx, row in model_input.iterrows():
            try:
                audio_base64 = row.get("audio_base64", row.get("audio", None))
                if audio_base64 is None:
                    raise ValueError("Missing audio_base64 column")
                
                audio_bytes = base64.b64decode(audio_base64)
                
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_bytes)
                    temp_path = temp_file.name
                
                try:
                    with torch.no_grad():
                        outputs = self.model.transcribe([temp_path])
                        transcription = outputs[0].text
                    
                    results.append({"transcription": transcription, "error": None})
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
            except Exception as e:
                results.append({"transcription": None, "error": str(e)[:500]})
        
        return pd.DataFrame(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Define Model Signature

# COMMAND ----------

input_schema = Schema([ColSpec("string", "audio_base64")])
output_schema = Schema([ColSpec("string", "transcription"), ColSpec("string", "error")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

print("Model Signature:")
print(f"  Inputs: {input_schema}")
print(f"  Outputs: {output_schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Test the Model Locally

# COMMAND ----------

print("Testing model locally before registration...")
test_model = ParakeetASRModel()

class MockContext:
    pass

test_model.load_context(MockContext())

# COMMAND ----------

print("Model loaded successfully!")
del test_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Log Model to MLflow / Unity Catalog

# COMMAND ----------

pip_requirements = [
    "torch>=2.0.0",
    "nemo_toolkit[asr]>=1.20.0",
    "pandas",
    "numpy",
]

# COMMAND ----------

print("=" * 80)
print("Registering Model with MLflow 3")
print("=" * 80)

experiment_name = f"/Users/{spark.sql('SELECT current_user()').collect()[0][0]}/parakeet_asr_experiment"
mlflow.set_experiment(experiment_name)
print(f"Experiment: {experiment_name}")

with mlflow.start_run(run_name="parakeet_asr_registration") as run:
    mlflow.log_params({
        "model_name": MODEL_NAME,
        "model_type": "ASR",
        "framework": "NeMo",
        "input_format": "base64_wav",
        "sample_rate": 16000,
    })
    
    print("Logging model to MLflow...")
    model_info = mlflow.pyfunc.log_model(
        artifact_path="parakeet_asr",
        python_model=ParakeetASRModel(),
        signature=signature,
        pip_requirements=pip_requirements,
        input_example=pd.DataFrame({"audio_base64": ["<base64_encoded_wav_audio>"]}),
        registered_model_name=UC_MODEL_NAME,
    )
    
    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    print(f"Model URI: {model_info.model_uri}")

print("=" * 80)
print("Model Registration Complete!")
print(f"Model registered to: {UC_MODEL_NAME}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("MODEL REGISTRATION COMPLETE")
print(f"Model Name: {UC_MODEL_NAME}")
print("Next: Run 00b_create_serving_endpoint notebook")