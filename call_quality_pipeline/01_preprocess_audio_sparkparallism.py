# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Audio Preprocessing
# MAGIC
# MAGIC This notebook discovers audio files in a Unity Catalog Volume, converts them to 16kHz mono WAV,
# MAGIC and splits long recordings into 5-minute chunks suitable for the serving endpoint's 16 MB payload limit.

# COMMAND ----------

# MAGIC %pip install pydub

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import *
from datetime import datetime, timezone

# COMMAND ----------

# Configuration
CATALOG = "<your-catalog>"
SCHEMA = "<your-schema>"
ROOT_FOLDER = f"/Volumes/{CATALOG}/{SCHEMA}/audios"
RUN_DATE = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
INTERMEDIATE_BASE = f"/Volumes/{CATALOG}/{SCHEMA}/parakeet_asr_temp/{RUN_DATE}"
PREPROCESSED_PATH = f"{INTERMEDIATE_BASE}/preprocessed"
NUM_FILES_TO_PROCESS = 1000

# CRITICAL: Chunk size to stay under 16 MB endpoint limit
# 5 min chunk @ 16kHz mono = ~9 MB WAV = ~12 MB base64 (under 16 MB limit)
MAX_CHUNK_MINUTES = 5.0  # DO NOT increase above 5 to avoid MAX_REQUEST_SIZE_EXCEEDED
CHUNK_OVERLAP_SECONDS = 1.5
NUM_PREPROCESSING_WORKERS = 16

print("=" * 80)
print("Preprocessing Configuration")
print("=" * 80)
print(f"Root Folder: {ROOT_FOLDER}")
print(f"Run Date: {RUN_DATE}")
print(f"Output: {PREPROCESSED_PATH}")
print(f"Max Chunk Size: {MAX_CHUNK_MINUTES} minutes (~{MAX_CHUNK_MINUTES * 1.92:.1f} MB per chunk)")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Discover Audio Files

# COMMAND ----------

all_files_df = (
    spark.read.format("binaryFile")
    .option("pathGlobFilter", "*.{mp3,wav}")
    .option("recursiveFileLookup", "true")
    .load(ROOT_FOLDER)
    .select("path", "length")
)

all_files_df = all_files_df.withColumn(
    "filename", F.element_at(F.split(F.col("path"), "/"), -1)
).withColumn(
    "parent_folder", F.element_at(F.split(F.col("path"), "/"), -2)
).withColumn(
    "full_path", F.col("path")
).withColumn(
    "file_id", F.regexp_replace(F.concat_ws("_", F.col("parent_folder"), F.col("filename")), "[/.]", "_")
)

total_files = all_files_df.count()
print(f"Found {total_files} audio files")
all_files_df.select("filename", "length").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Output Directories

# COMMAND ----------

# Create the volume for temporary/intermediate files
spark.sql(f"""
CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.parakeet_asr_temp
COMMENT 'Temporary storage for Parakeet ASR preprocessing'
""")
print(f"Volume {CATALOG}.{SCHEMA}.parakeet_asr_temp created/verified")
print(f"Created: {PREPROCESSED_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocess Audio Files

# COMMAND ----------

import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

result_schema = StructType([
    StructField("parent_folder", StringType(), False),
    StructField("filename", StringType(), False),
    StructField("full_path", StringType(), False),
    StructField("file_id", StringType(), False),
    StructField("duration_seconds", DoubleType(), True),
    StructField("num_chunks", IntegerType(), True),
    StructField("chunk_paths", StringType(), True),
    StructField("error_message", StringType(), True),
])

def preprocess_partition(iterator):
    import os
    from pydub import AudioSegment
    
    # 5 min chunks to stay under 16 MB endpoint limit
    max_chunk_ms = 5.0 * 60 * 1000  # 5 minutes in milliseconds
    overlap_ms = 1.5 * 1000
    preprocessed_path = PREPROCESSED_PATH.replace("dbfs:", "")
    
    os.makedirs(preprocessed_path, exist_ok=True)
    
    for pdf in iterator:
        results = []
        for _, row in pdf.iterrows():
            try:
                source_path = row["full_path"].replace("dbfs:", "")
                
                if source_path.endswith(".mp3"):
                    audio = AudioSegment.from_mp3(source_path)
                else:
                    audio = AudioSegment.from_file(source_path)
                audio = audio.set_frame_rate(16000).set_channels(1)
                duration = audio.duration_seconds
                
                chunk_paths = []
                if len(audio) <= max_chunk_ms:
                    wav_filename = f"{row['file_id']}_chunk0.wav"
                    wav_path = os.path.join(preprocessed_path, wav_filename)
                    audio.export(wav_path, format="wav")
                    dbfs_path = f"{PREPROCESSED_PATH}/{wav_filename}"
                    chunk_paths.append(dbfs_path)
                else:
                    num_chunks = int(len(audio) / (max_chunk_ms - overlap_ms)) + 1
                    for i in range(num_chunks):
                        start = max(0, int(i * (max_chunk_ms - overlap_ms)))
                        end = min(len(audio), int(start + max_chunk_ms))
                        chunk = audio[start:end]
                        wav_filename = f"{row['file_id']}_chunk{i}.wav"
                        wav_path = os.path.join(preprocessed_path, wav_filename)
                        chunk.export(wav_path, format="wav")
                        dbfs_path = f"{PREPROCESSED_PATH}/{wav_filename}"
                        chunk_paths.append(dbfs_path)
                
                results.append({
                    "parent_folder": row["parent_folder"],
                    "filename": row["filename"],
                    "full_path": row["full_path"],
                    "file_id": row["file_id"],
                    "duration_seconds": duration,
                    "num_chunks": len(chunk_paths),
                    "chunk_paths": "|".join(chunk_paths),
                    "error_message": None,
                })
            except Exception as e:
                results.append({
                    "parent_folder": row["parent_folder"],
                    "filename": row["filename"],
                    "full_path": row["full_path"],
                    "file_id": row["file_id"],
                    "duration_seconds": None,
                    "num_chunks": None,
                    "chunk_paths": None,
                    "error_message": str(e)[:500],
                })
        yield pd.DataFrame(results)

# COMMAND ----------

import time
start = time.time()

selected_df = all_files_df.limit(NUM_FILES_TO_PROCESS)
processed_df = selected_df.repartition(NUM_PREPROCESSING_WORKERS).mapInPandas(preprocess_partition, schema=result_schema) 
processed_df = processed_df.cache()

total = processed_df.count()
success = processed_df.filter(F.col("error_message").isNull()).count()
failed = total - success

print(f"Preprocessed: {success} successful, {failed} failed")
print(f"Time: {(time.time() - start)/60:.2f} min")

# COMMAND ----------

# Save metadata
metadata_table = f"{CATALOG}.{SCHEMA}.preprocessing_metadata_{RUN_DATE}"
processed_df.filter(F.col("error_message").isNull()).write.format("delta").mode("overwrite").saveAsTable(metadata_table)
print(f"Saved to: {metadata_table}")

processed_df.select("filename", "duration_seconds", "num_chunks").show(10, truncate=False)