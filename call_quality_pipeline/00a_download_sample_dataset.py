# Databricks notebook source
# MAGIC %md
# MAGIC # 00a Download Sample Dataset
# MAGIC
# MAGIC This notebook downloads the Appen 1000h US English Smartphone Conversation dataset
# MAGIC from Hugging Face and saves the audio files to a Unity Catalog Volume for use with
# MAGIC the rest of the pipeline.

# COMMAND ----------

# MAGIC %pip install datasets

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "<your-catalog>"
SCHEMA = "<your-schema>"
VOLUME_NAME = "audios"
VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME_NAME}"

DATASET_ID = "Appenlimited/1000h-us-english-smartphone-conversation"

print("=" * 80)
print("Sample Dataset Download")
print("=" * 80)
print(f"Dataset: {DATASET_ID}")
print(f"Target Volume: {VOLUME_PATH}")
print("=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Schema and Volume

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"""
CREATE VOLUME IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`.`{VOLUME_NAME}`
COMMENT 'Audio files for call quality pipeline'
""")
print(f"Volume ready: {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download and Save Audio Files

# COMMAND ----------

import os
import wave
import shutil
import tempfile
import numpy as np
from datasets import load_dataset

os.makedirs(VOLUME_PATH, exist_ok=True)

ds = load_dataset(DATASET_ID, split="train")
print(f"Dataset loaded: {len(ds)} recordings")

# Write to local temp dir first, then copy to Volume
# (libsndfile/soundfile cannot write directly to FUSE-mounted UC Volumes)
tmp_dir = tempfile.mkdtemp()

for i, row in enumerate(ds):
    audio = row["audio"]
    arr = np.clip(np.array(audio["array"]) * 32767, -32768, 32767).astype(np.int16)
    sr = audio["sampling_rate"]

    filename = f"conversation_{i:03d}.wav"
    tmp_path = os.path.join(tmp_dir, filename)
    vol_path = os.path.join(VOLUME_PATH, filename)

    with wave.open(tmp_path, "w") as wf:
        wf.setnchannels(1 if arr.ndim == 1 else arr.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(arr.tobytes())

    shutil.copy2(tmp_path, vol_path)
    os.remove(tmp_path)
    print(f"  [{i+1}/{len(ds)}] Saved {filename} ({sr} Hz)")

shutil.rmtree(tmp_dir)
print(f"\nSaved {len(ds)} audio files to {VOLUME_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Downloaded Files

# COMMAND ----------

files = [f for f in os.listdir(VOLUME_PATH) if f.endswith(".wav")]
total_size_mb = sum(os.path.getsize(os.path.join(VOLUME_PATH, f)) for f in files) / (1024 * 1024)

print("=" * 80)
print("DOWNLOAD COMPLETE")
print("=" * 80)
print(f"Files: {len(files)} WAV recordings")
print(f"Total size: {total_size_mb:.1f} MB")
print(f"Location: {VOLUME_PATH}")
print("=" * 80)
print("Next: Run 00_register_model.py to register the Parakeet ASR model")
print("=" * 80)
