# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Transcribe Audio via Serving Endpoint
# MAGIC
# MAGIC This notebook sends preprocessed WAV chunks to the Parakeet ASR serving endpoint
# MAGIC using Spark-distributed workers and reassembles the transcriptions per file.

# COMMAND ----------

# MAGIC %pip install requests

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import time
import base64
import requests
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timezone
from databricks.sdk import WorkspaceClient

# COMMAND ----------

# Configuration
CATALOG = "<your-catalog>"
SCHEMA = "<your-schema>"
RUN_DATE = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
ENDPOINT_NAME = "parakeet-asr-endpoint"
WORKSPACE_URL = "<your-workspace-url>"
NUM_WORKERS = 8

# Payload size limits (Databricks Model Serving)
MAX_PAYLOAD_SIZE_MB = 16
MAX_PAYLOAD_SIZE_BYTES = MAX_PAYLOAD_SIZE_MB * 1024 * 1024
MAX_FILE_SIZE_BYTES = int(MAX_PAYLOAD_SIZE_BYTES / 1.34)

print("=" * 80)
print("Transcription Configuration")
print("=" * 80)
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"Run Date: {RUN_DATE}")
print(f"Max payload size: {MAX_PAYLOAD_SIZE_MB} MB")
print(f"Max file size (pre-encoding): {MAX_FILE_SIZE_BYTES / (1024*1024):.1f} MB")
print("=" * 80)

# COMMAND ----------

def warm_up_endpoint(endpoint_name, workspace_url, timeout_minutes=5):
    token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    base_url = workspace_url.rstrip("/")
    if not base_url.startswith("https://"):
        base_url = f"https://{base_url}"
    endpoint_url = f"{base_url}/serving-endpoints/{endpoint_name}/invocations"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    import wave, struct, tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
    with wave.open(temp_path, 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack('<' + 'h' * 16000, *([0] * 16000)))
    with open(temp_path, 'rb') as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')
    payload = {"dataframe_records": [{"audio_base64": audio_b64}]}
    print(f"Warming up endpoint '{endpoint_name}'...")
    start = time.time()
    try:
        response = requests.post(endpoint_url, headers=headers, json=payload, timeout=timeout_minutes*60)
        elapsed = time.time() - start
        if response.status_code == 200:
            print(f"Endpoint ready! (took {elapsed:.1f}s)")
            return True
        else:
            print(f"Warning: Status {response.status_code} - {response.text[:200]}")
            return False
    except Exception as e:
        print(f"Error warming up: {e}")
        return False

warm_up_endpoint(ENDPOINT_NAME, WORKSPACE_URL)

# COMMAND ----------

metadata_table = f"{CATALOG}.{SCHEMA}.preprocessing_metadata_{RUN_DATE}"
metadata_df = spark.table(metadata_table)
print(f"Loaded {metadata_df.count()} files to transcribe")
metadata_df.select("filename", "duration_seconds", "num_chunks").show(5)

# COMMAND ----------

result_schema = StructType([
    StructField("parent_folder", StringType(), False),
    StructField("filename", StringType(), False),
    StructField("full_path", StringType(), False),
    StructField("file_id", StringType(), False),
    StructField("duration_seconds", DoubleType(), True),
    StructField("num_chunks", IntegerType(), True),
    StructField("transcription_text", StringType(), True),
    StructField("processing_timestamp", TimestampType(), True),
    StructField("error_message", StringType(), True),
])

# COMMAND ----------

api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
workspace_url = WORKSPACE_URL
endpoint_name = ENDPOINT_NAME
max_file_size = MAX_FILE_SIZE_BYTES
max_payload_size = MAX_PAYLOAD_SIZE_BYTES

def transcribe_partition(iterator):
    import os, json, base64, requests
    from datetime import datetime, timezone
    base_url = workspace_url.rstrip("/")
    if not base_url.startswith("https://"):
        base_url = f"https://{base_url}"
    endpoint_url = f"{base_url}/serving-endpoints/{endpoint_name}/invocations"
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}
    for pdf in iterator:
        results = []
        for _, row in pdf.iterrows():
            chunk_errors = []
            try:
                chunk_paths = row["chunk_paths"].split("|")
                texts = []
                for chunk_idx, chunk_path in enumerate(chunk_paths):
                    dbfs_path = chunk_path.replace("dbfs:", "/dbfs")
                    file_size = os.path.getsize(dbfs_path)
                    file_size_mb = file_size / (1024 * 1024)
                    if file_size > max_file_size:
                        chunk_errors.append(f"CHUNK_TOO_LARGE: Chunk {chunk_idx + 1}/{len(chunk_paths)} is {file_size_mb:.2f} MB")
                        continue
                    with open(dbfs_path, "rb") as f:
                        audio_bytes = f.read()
                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    payload = {"dataframe_records": [{"audio_base64": audio_b64}]}
                    payload_size = len(json.dumps(payload).encode('utf-8'))
                    payload_size_mb = payload_size / (1024 * 1024)
                    if payload_size > max_payload_size:
                        chunk_errors.append(f"PAYLOAD_TOO_LARGE: {payload_size_mb:.2f} MB")
                        continue
                    resp = requests.post(endpoint_url, headers=headers, json=payload, timeout=300)
                    if resp.status_code == 200:
                        result = resp.json()
                        if "predictions" in result:
                            texts.append(result["predictions"][0].get("transcription", ""))
                        elif "dataframe_records" in result:
                            texts.append(result["dataframe_records"][0].get("transcription", ""))
                    else:
                        try:
                            error_body = resp.json()
                            error_code = error_body.get("error_code", "UNKNOWN")
                            error_message = error_body.get("message", resp.text[:200])
                        except Exception:
                            error_code = f"HTTP_{resp.status_code}"
                            error_message = resp.text[:200]
                        chunk_errors.append(f"ENDPOINT_ERROR: {error_code} - {error_message[:100]}")
                final_error = "; ".join(chunk_errors) if chunk_errors else None
                results.append({
                    "parent_folder": row["parent_folder"], "filename": row["filename"],
                    "full_path": row["full_path"], "file_id": row["file_id"],
                    "duration_seconds": row["duration_seconds"], "num_chunks": row["num_chunks"],
                    "transcription_text": " ".join(texts) if texts else None,
                    "processing_timestamp": datetime.now(tz=timezone.utc),
                    "error_message": final_error[:1000] if final_error else None,
                })
            except Exception as e:
                results.append({
                    "parent_folder": row["parent_folder"], "filename": row["filename"],
                    "full_path": row["full_path"], "file_id": row["file_id"],
                    "duration_seconds": row["duration_seconds"], "num_chunks": row["num_chunks"],
                    "transcription_text": None, "processing_timestamp": datetime.now(tz=timezone.utc),
                    "error_message": f"EXCEPTION: {str(e)[:500]}",
                })
        yield pd.DataFrame(results)

start = time.time()
transcribed_df = metadata_df.repartition(NUM_WORKERS).mapInPandas(transcribe_partition, schema=result_schema)
transcribed_df = transcribed_df.cache()
total = transcribed_df.count()
success = transcribed_df.filter(F.col("error_message").isNull()).count()
print(f"Transcribed: {success}/{total}")
print(f"Time: {(time.time() - start)/60:.2f} min")

# COMMAND ----------

result_table = f"{CATALOG}.{SCHEMA}.transcription_results_{RUN_DATE}"
transcribed_df.write.format("delta").mode("overwrite").saveAsTable(result_table)
print(f"Saved to: {result_table}")
transcribed_df.select("filename", F.substring("transcription_text", 1, 100).alias("preview")).show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Error Analysis Helper

# COMMAND ----------

def analyze_transcription_errors(result_table_name):
    df = spark.table(result_table_name)
    total = df.count()
    errors_df = df.filter(F.col("error_message").isNotNull())
    error_count = errors_df.count()
    success_count = total - error_count
    print("=" * 80)
    print("TRANSCRIPTION ERROR ANALYSIS")
    print("=" * 80)
    print(f"Total: {total}, Success: {success_count}, Errors: {error_count}")
    if error_count > 0:
        error_categories = errors_df.withColumn("error_type", 
            F.when(F.col("error_message").contains("CHUNK_TOO_LARGE"), "CHUNK_TOO_LARGE")
            .when(F.col("error_message").contains("PAYLOAD_TOO_LARGE"), "PAYLOAD_TOO_LARGE")
            .when(F.col("error_message").contains("MAX_REQUEST_SIZE_EXCEEDED"), "MAX_REQUEST_SIZE_EXCEEDED")
            .otherwise("OTHER")
        ).groupBy("error_type").count().orderBy(F.desc("count"))
        error_categories.show()
        errors_df.select("filename", "error_message").show(5, truncate=100)
    return df

# analyze_transcription_errors(f"{CATALOG}.{SCHEMA}.transcription_results_{RUN_DATE}")