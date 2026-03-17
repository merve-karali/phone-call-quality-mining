# Databricks notebook source
# MAGIC %md
# MAGIC # 04 AI Analysis & Evaluation
# MAGIC
# MAGIC This notebook applies AI-powered analytics on transcribed call data using Databricks AI Functions
# MAGIC (`ai_analyze_sentiment`, `ai_summarize`, `ai_query`) and evaluates output quality using MLflow GenAI evaluation.

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow[databricks]>=3.1.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

CATALOG = "<your-catalog>"
SCHEMA = "<your-schema>"
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.call_transcriptions"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.enriched_transcriptions"

# Any pay-per-token or provisioned throughput Foundation Model endpoint
ENDPOINT_NAME = "databricks-meta-llama-3-3-70b-instruct"

print(f"Source: {SOURCE_TABLE}")
print(f"Output: {GOLD_TABLE}")
print(f"Endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

df = spark.table(SOURCE_TABLE)
record_count = df.count()

if record_count == 0:
    dbutils.notebook.exit("No records to process. Exiting.")

print(f"Loaded {record_count} transcriptions")
display(df.select("filename", "transcription_text").limit(5))

# COMMAND ----------

prompt = """Analyze the following phone conversation transcript between two speakers.

Your task:
1. Identify the speakers by name if mentioned, otherwise label them Speaker 1 and Speaker 2
2. Determine the main topics discussed
3. Assess the overall tone of the conversation
4. Provide a brief summary

Phone Conversation Transcript:
"""

response_format = """{
    "type": "json_schema",
    "json_schema": {
        "name": "conversation_analysis",
        "schema": {
            "type": "object",
            "properties": {
                "speaker_1": {
                    "type": "string",
                    "description": "Name of the first speaker if mentioned, otherwise Speaker 1"
                },
                "speaker_2": {
                    "type": "string",
                    "description": "Name of the second speaker if mentioned, otherwise Speaker 2"
                },
                "main_topics": {
                    "type": "string",
                    "description": "Comma-separated list of main topics discussed in the conversation"
                },
                "conversation_tone": {
                    "type": "string",
                    "description": "Overall tone of the conversation (e.g. friendly, formal, argumentative, casual)"
                },
                "conversation_summary": {
                    "type": "string",
                    "description": "A brief summary of the conversation"
                }
            },
            "required": [
                "speaker_1",
                "speaker_2",
                "main_topics",
                "conversation_tone",
                "conversation_summary"
            ]
        },
        "strict": true
    }
}"""

# COMMAND ----------

df.createOrReplaceTempView("transcriptions_temp")

query = f"""
SELECT *,
       ai_analyze_sentiment(transcription_text) AS sentiment,
       ai_summarize(transcription_text) AS summary,
       ai_query('{ENDPOINT_NAME}', CONCAT('{prompt}', transcription_text), responseFormat => '{response_format}') AS conversation_analysis
FROM transcriptions_temp
WHERE transcription_text IS NOT NULL
"""

enriched_df = spark.sql(query)
display(enriched_df.select("filename", "sentiment", "summary").limit(5))

# COMMAND ----------

from pyspark.sql.functions import col, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType

analysis_schema = StructType([
    StructField("speaker_1", StringType(), True),
    StructField("speaker_2", StringType(), True),
    StructField("main_topics", StringType(), True),
    StructField("conversation_tone", StringType(), True),
    StructField("conversation_summary", StringType(), True)
])

gold_df = enriched_df \
    .withColumn("analysis_parsed", from_json(col("conversation_analysis"), analysis_schema)) \
    .withColumn("speaker_1", col("analysis_parsed.speaker_1")) \
    .withColumn("speaker_2", col("analysis_parsed.speaker_2")) \
    .withColumn("main_topics", col("analysis_parsed.main_topics")) \
    .withColumn("conversation_tone", col("analysis_parsed.conversation_tone")) \
    .withColumn("conversation_summary", col("analysis_parsed.conversation_summary")) \
    .withColumn("enriched_timestamp", current_timestamp()) \
    .drop("conversation_analysis", "analysis_parsed")

display(gold_df.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow GenAI Evaluation

# COMMAND ----------

import mlflow
import pandas as pd

current_user = spark.sql('SELECT current_user()').collect()[0][0]
experiment_name = f"/Users/{current_user}/call_analysis_evaluation"
mlflow.set_experiment(experiment_name)
print(f"MLflow Experiment: {experiment_name}")

# COMMAND ----------

eval_df = gold_df.select(
    col("transcription_text"),
    col("summary").alias("outputs")
).limit(10).toPandas()

# Convert 'inputs' to required dict format
eval_df["inputs"] = eval_df["transcription_text"].apply(
    lambda x: {"transcript": x}
)
eval_df = eval_df.drop(columns=["transcription_text"])

print(f"Prepared {len(eval_df)} evaluation records")

# COMMAND ----------

from mlflow.genai.scorers import Guidelines, Safety

scorers = [
    Guidelines(
        guidelines="The summary accurately captures the main topics and key points of the phone conversation",
        name="captures_main_topics",
    ),
    Safety(),
]

print(f"Defined {len(scorers)} scorers")

# COMMAND ----------

with mlflow.start_run(run_name="call_summary_evaluation"):
    mlflow.log_param("endpoint", ENDPOINT_NAME)
    mlflow.log_param("num_samples", len(eval_df))
    mlflow.log_param("source_table", SOURCE_TABLE)
    
    results = mlflow.genai.evaluate(
        data=eval_df,
        scorers=scorers,
    )
    
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, value in results.metrics.items():
        print(f"  {metric}: {value}")
    print("=" * 50)

# COMMAND ----------

if not spark.catalog.tableExists(GOLD_TABLE):
    gold_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(GOLD_TABLE)
else:
    gold_df.write.mode("append").saveAsTable(GOLD_TABLE)

print(f"AI enriched insights written to {GOLD_TABLE}")

# COMMAND ----------

print("\nSentiment Distribution:")
spark.sql(f"SELECT sentiment, COUNT(*) as count FROM {GOLD_TABLE} GROUP BY sentiment ORDER BY count DESC").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Output
# MAGIC **Table: enriched_transcriptions**
# MAGIC
# MAGIC Includes sentiment, summaries, speaker identification, and evaluation metrics.