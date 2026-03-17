# Databricks notebook source
# MAGIC %md
# MAGIC # Create Model Serving Endpoint for Parakeet ASR
# MAGIC
# MAGIC This notebook:
# MAGIC 1. Creates a GPU-enabled model serving endpoint
# MAGIC 2. Configures scale-to-zero for cost optimization (ideal for batch processing)
# MAGIC 3. Provides utilities for endpoint management

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

# Import error classes with fallback for older SDK versions
try:
    from databricks.sdk.errors import NotFound, ResourceDoesNotExist
except ImportError:
    # Fallback for older SDK versions
    NotFound = type('NotFound', (Exception,), {})
    ResourceDoesNotExist = type('ResourceDoesNotExist', (Exception,), {})

# Configuration - inline values (no external imports)
CATALOG = "<your-catalog>"
SCHEMA = "<your-schema>"
WORKSPACE_URL = "<your-workspace-url>"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Initialize Databricks SDK client
w = WorkspaceClient()

# Endpoint configuration
ENDPOINT_NAME = "parakeet-asr-endpoint"
UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.parakeet_asr_model"

# Serving configuration
WORKLOAD_SIZE = "Small"
WORKLOAD_TYPE = ServingModelWorkloadType.GPU_SMALL

# Scale-to-zero configuration
SCALE_TO_ZERO_ENABLED = True

print("=" * 80)
print("Serving Endpoint Configuration")
print("=" * 80)
print(f"Endpoint Name: {ENDPOINT_NAME}")
print(f"Model: {UC_MODEL_NAME}")
print(f"Workload Type: {WORKLOAD_TYPE}")
print(f"Scale to Zero: {SCALE_TO_ZERO_ENABLED}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Check if Endpoint Already Exists

# COMMAND ----------

def get_endpoint_status(endpoint_name: str) -> dict:
    """Get the current status of a serving endpoint."""
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
        return {
            "exists": True,
            "state": endpoint.state.ready,
            "config_update": endpoint.state.config_update,
            "endpoint": endpoint,
        }
    except (NotFound, ResourceDoesNotExist):
        return {"exists": False}
    except Exception as e:
        # Fallback: Handle "RESOURCE_DOES_NOT_EXIST" in error message
        error_str = str(e)
        if "RESOURCE_DOES_NOT_EXIST" in error_str or "does not exist" in error_str.lower():
            return {"exists": False}
        raise

status = get_endpoint_status(ENDPOINT_NAME)
if status["exists"]:
    print(f"Endpoint '{ENDPOINT_NAME}' already exists")
    print(f"State: {status['state']}")
else:
    print(f"Endpoint '{ENDPOINT_NAME}' does not exist - will create")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create the Endpoint

# COMMAND ----------

# Get the model version number for the 'champion' alias
import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# Try to get version from 'champion' alias, fallback to latest version
try:
    model_version_info = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
    MODEL_VERSION = model_version_info.version
    print(f"Using 'champion' alias -> version {MODEL_VERSION}")
except Exception as e:
    # Fallback: get the latest version
    versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
    if versions:
        MODEL_VERSION = str(max(int(v.version) for v in versions))
        print(f"No 'champion' alias found, using latest version {MODEL_VERSION}")
    else:
        raise ValueError(f"No versions found for model {UC_MODEL_NAME}")

# COMMAND ----------

if not status["exists"]:
    # Configure served entity (the model)
    served_entity = ServedEntityInput(
        entity_name=UC_MODEL_NAME,
        entity_version=MODEL_VERSION,
        workload_size="Small",
        workload_type=WORKLOAD_TYPE,
        scale_to_zero_enabled=SCALE_TO_ZERO_ENABLED,
    )
    
    # Configure endpoint
    endpoint_config = EndpointCoreConfigInput(
        served_entities=[served_entity],
    )    
    print(f"Creating endpoint '{ENDPOINT_NAME}'...")
    print(f"  Model: {UC_MODEL_NAME}")
    print(f"  Workload Type: {WORKLOAD_TYPE}")
    print(f"  Scale to Zero: {SCALE_TO_ZERO_ENABLED}")
    
    # Create the endpoint
    endpoint = w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=endpoint_config,
    )
    
    print("Endpoint creation initiated!")
else:
    print(f"Endpoint '{ENDPOINT_NAME}' already exists.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Wait for Endpoint to be Ready

# COMMAND ----------

def wait_for_endpoint_ready(endpoint_name: str, timeout_minutes: int = 30):
    """Wait for the serving endpoint to be ready."""
    
    print(f"Waiting for endpoint '{endpoint_name}' to be ready...")
    print("This may take 10-15 minutes for GPU endpoints...")
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while True:
        status = get_endpoint_status(endpoint_name)
        
        if not status["exists"]:
            print("ERROR: Endpoint does not exist")
            return False
        
        state = status["state"]
        elapsed = time.time() - start_time
        
        # Convert state to string for comparison (handles enum values)
        state_str = str(state)
        print(f"  [{elapsed/60:.1f}min] State: {state_str}")
        
        if "READY" in state_str:
            print(f"Endpoint is READY! (took {elapsed/60:.1f} minutes)")
            return True
        
        if elapsed > timeout_seconds:
            print(f"TIMEOUT: Endpoint not ready after {timeout_minutes} minutes")
            return False
        
        if "FAILED" in state_str:
            print("ERROR: Endpoint deployment failed")
            return False
        
        time.sleep(30)

# COMMAND ----------

# Wait for endpoint to be ready
if not status["exists"]:
    is_ready = wait_for_endpoint_ready(ENDPOINT_NAME, timeout_minutes=30)
else:
    # Check if already ready
    current_status = get_endpoint_status(ENDPOINT_NAME)
    state_str = str(current_status.get("state", ""))
    is_ready = "READY" in state_str
    if is_ready:
        print(f"Endpoint '{ENDPOINT_NAME}' is already READY")
    else:
        print(f"Endpoint state: {state_str}")
        is_ready = wait_for_endpoint_ready(ENDPOINT_NAME, timeout_minutes=30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Enable AI Gateway

# COMMAND ----------

import requests

# Get workspace URL and token
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
API_URL = WORKSPACE_URL.rstrip("/")
if not API_URL.startswith("https://"):
    API_URL = f"https://{API_URL}"

# AI Gateway configuration
RATE_LIMIT_CALLS = 1000  # calls per minute

ai_gateway_payload = {
    "usage_tracking_config": {
        "enabled": True
    },
    "inference_table_config": {
        "catalog_name": CATALOG,
        "schema_name": SCHEMA,
        "table_name_prefix": "parakeet_inference",
        "enabled": True
    },
    "rate_limits": [
        {
            "calls": RATE_LIMIT_CALLS,
            "renewal_period": "minute",
            "key": "endpoint"
        }
    ]
}

print(f"Enabling AI Gateway on '{ENDPOINT_NAME}'...")

response = requests.put(
    f"{API_URL}/api/2.0/serving-endpoints/{ENDPOINT_NAME}/ai-gateway",
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json=ai_gateway_payload
)

if response.status_code == 200:
    print("AI Gateway enabled successfully!")
    print(f"  - Usage Tracking: Enabled")
    print(f"  - Inference Tables: {CATALOG}.{SCHEMA}.parakeet_inference_*")
    print(f"  - Rate Limit: {RATE_LIMIT_CALLS} calls/minute")
else:
    print(f"Warning: AI Gateway returned {response.status_code}")
    print(response.text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("=" * 80)
print("SERVING ENDPOINT SETUP COMPLETE")
print("=" * 80)
print(f"Endpoint Name: {ENDPOINT_NAME}")
print(f"Model: {UC_MODEL_NAME}")
print(f"Scale to Zero: {SCALE_TO_ZERO_ENABLED}")
print(f"Status: {'READY' if is_ready else 'PENDING'}")
print("=" * 80)
print("Next Steps:")
print("1. Run '01_preprocess_audio_sparkparallism.py' to prepare audio files")
print("2. The endpoint will scale to zero when idle to save costs")
print("3. First request after scale-down will have ~2-3 min cold start")
print("=" * 80)