# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Merge Transcriptions to Delta Table
# MAGIC
# MAGIC This notebook merges transcription results into a persistent Delta table using
# MAGIC idempotent MERGE INTO, then cleans up intermediate tables and files.

# COMMAND ----------

from pyspark.sql import functions as F
from datetime import datetime, timezone

# COMMAND ----------

# Configuration
CATALOG = "<your-catalog>"
SCHEMA = "<your-schema>"
TABLE_NAME = "call_transcriptions"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE_NAME}"
RUN_DATE = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
CLEANUP_INTERMEDIATE = True

print("=" * 80)
print("Merge Configuration")
print("=" * 80)
print(f"Target Table: {FULL_TABLE_NAME}")
print(f"Run Date: {RUN_DATE}")
print("=" * 80)

# COMMAND ----------

# Load transcription results
result_table = f"{CATALOG}.{SCHEMA}.transcription_results_{RUN_DATE}"
transcriptions_df = spark.table(result_table)
transcriptions_df = transcriptions_df.withColumn("ingestion_timestamp", F.current_timestamp())

record_count = transcriptions_df.count()
print(f"Loaded {record_count} transcriptions")

# COMMAND ----------

# Create target table if not exists
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

create_sql = f"""
CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME} (
    parent_folder STRING NOT NULL,
    filename STRING NOT NULL,
    full_path STRING NOT NULL,
    duration_seconds DOUBLE,
    num_chunks INT,
    transcription_text STRING,
    processing_timestamp TIMESTAMP,
    ingestion_timestamp TIMESTAMP NOT NULL
)
USING DELTA
CLUSTER BY (parent_folder, filename)
"""
spark.sql(create_sql)
print(f"Table {FULL_TABLE_NAME} ready")

# COMMAND ----------

# Merge into target table
transcriptions_df.createOrReplaceTempView("new_transcriptions")

merge_sql = f"""
MERGE INTO {FULL_TABLE_NAME} as target
USING new_transcriptions as source
ON target.parent_folder = source.parent_folder AND target.filename = source.filename
WHEN MATCHED THEN UPDATE SET
    target.full_path = source.full_path,
    target.duration_seconds = source.duration_seconds,
    target.num_chunks = source.num_chunks,
    target.transcription_text = source.transcription_text,
    target.processing_timestamp = source.processing_timestamp,
    target.ingestion_timestamp = source.ingestion_timestamp
WHEN NOT MATCHED THEN INSERT *
"""

spark.sql(merge_sql)
new_count = spark.sql(f"SELECT COUNT(*) FROM {FULL_TABLE_NAME}").collect()[0][0]
print(f"Table now has {new_count} records")

# COMMAND ----------

# Show results
spark.sql(f"""
SELECT filename, duration_seconds, LEFT(transcription_text, 100) as preview
FROM {FULL_TABLE_NAME}
ORDER BY ingestion_timestamp DESC
LIMIT 5
""").show(truncate=False)

# COMMAND ----------

# Cleanup intermediate tables
if CLEANUP_INTERMEDIATE:
    tables = [
        f"{CATALOG}.{SCHEMA}.preprocessing_metadata_{RUN_DATE}",
        f"{CATALOG}.{SCHEMA}.transcription_results_{RUN_DATE}",
    ]
    for t in tables:
        try:
            if spark.catalog.tableExists(t):
                spark.sql(f"DROP TABLE IF EXISTS {t}")
                print(f"Dropped: {t}")
        except:
            pass
    
    # Cleanup intermediate files
    try:
        dbutils.fs.rm(f"/Volumes/{CATALOG}/{SCHEMA}/parakeet_asr_temp/{RUN_DATE}", recurse=True)
        print("Cleaned up intermediate files")
    except:
        pass

# COMMAND ----------

print("=" * 80)
print("PIPELINE COMPLETE")
print("=" * 80)
print(f"Table: {FULL_TABLE_NAME}")
print(f"Total Records: {new_count}")
print("=" * 80)