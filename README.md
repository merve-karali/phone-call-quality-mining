# Phone Call Quality Mining

End-to-end Databricks pipeline for transcribing phone calls and extracting quality insights using AI.

## Use Case

Organizations handle thousands of phone conversations daily -- customer service calls, sales calls, internal discussions -- but only a fraction are ever reviewed. Manual listening is slow, expensive, and unscalable.

This pipeline automates the entire workflow:

1. **Transcribe** audio recordings at scale using NVIDIA Parakeet ASR on a GPU serving endpoint
2. **Analyze** every transcript with Databricks AI Functions for sentiment, summarization, and speaker identification
3. **Evaluate** output quality using MLflow 3 GenAI scorers to ensure reliability

The result is a structured Delta table of enriched call insights -- sentiment scores, conversation summaries, speaker roles, and customer inquiries -- ready for dashboarding, compliance review, or targeted agent training.

**Applicable to:** customer service quality monitoring, sales call review, compliance auditing, agent performance analytics, and any scenario requiring insights from recorded conversations.

## Architecture

```
Raw Audio Files          NVIDIA Parakeet         Databricks AI Functions
(Databricks Volume)  →   ASR Endpoint      →    (Sentiment, Summary,
     MP3/WAV           (Model Serving)           Speaker Identification)
                              ↓                          ↓
                     Transcriptions Table        Enriched Insights Table
                        (Delta Lake)                (Delta Lake)
```

## Pipeline Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 00a | `00a_download_sample_dataset.py` | Download sample audio dataset from Hugging Face to a Unity Catalog Volume |
| 00 | `00_register_model.py` | Register NVIDIA Parakeet TDT 0.6B v3 ASR model in Unity Catalog via MLflow 3 |
| 00b | `00b_create_serving_endpoint.py` | Create GPU model serving endpoint with scale-to-zero and AI Gateway |
| 01 | `01_preprocess_audio_sparkparallism.py` | Spark-parallelized audio preprocessing: MP3/WAV → 16kHz mono WAV, chunked to 5-min segments (Parakeet requires WAV input) |
| 02 | `02_transcribe_audio.py` | Distributed transcription via serving endpoint using Spark `mapInPandas` |
| 03 | `03_merge_to_table.py` | Merge transcriptions into a Delta table with idempotent MERGE INTO |
| 04 | `04_ai_analysis.py` | AI-powered analysis using `ai_analyze_sentiment`, `ai_summarize`, `ai_query` + MLflow GenAI evaluation |

## Configuration

Before running, update the following placeholders in each notebook:

```python
CATALOG = "<your-catalog>"          # Your Unity Catalog catalog name
SCHEMA = "<your-schema>"             # Your Unity Catalog schema name
WORKSPACE_URL = "<your-workspace-url>"  # e.g. adb-xxxxx.xx.azuredatabricks.net
```

## Sample Dataset

This pipeline uses the [Appen 1000h US English Smartphone Conversation](https://huggingface.co/datasets/Appenlimited/1000h-us-english-smartphone-conversation) dataset -- 40 real two-speaker American English conversations (~400 MB) freely available on Hugging Face. The recordings capture natural phone conversations between pairs of speakers, making them a suitable proxy for customer service or internal call recordings.

**Download via Python (run on Databricks):**

Or simply run notebook `00a_download_sample_dataset.py` which handles everything automatically.

**Dataset details:**
- **Source:** [Appenlimited/1000h-us-english-smartphone-conversation](https://huggingface.co/datasets/Appenlimited/1000h-us-english-smartphone-conversation)
- **Format:** WAV audio, ~10 MB per file
- **Content:** 40 real two-speaker American English conversations with transcriptions and speaker metadata

## Databricks Components

### Unity Catalog
- **Model Registry** -- Parakeet ASR model registered as a versioned UC model (`<catalog>.<schema>.parakeet_asr_model`)
- **Managed Tables** -- Delta tables for transcriptions and enriched insights with schema enforcement
- **Volumes** -- File storage for raw audio files and intermediate WAV chunks

### Model Serving
- **GPU Serving Endpoint** -- Custom PyFunc model deployed on `GPU_SMALL` workload type
- **Scale-to-Zero** -- Endpoint automatically scales down when idle and warms up on demand
- **AI Gateway** -- Inference table logging, usage tracking, and rate limiting enabled on the endpoint

### AI Functions (SQL)
- `ai_analyze_sentiment()` -- Sentiment scoring on each transcript
- `ai_summarize()` -- Concise call summaries
- `ai_query()` -- Structured speaker identification (agent name, customer name, role, inquiry, conversation summary) via LLM with JSON schema enforcement

### MLflow 3
- **PyFunc Model Wrapper** -- Custom `mlflow.pyfunc.PythonModel` for Parakeet ASR inference
- **Model Logging** -- Model artifact, signature, and pip requirements logged to UC
- **GenAI Evaluation** -- `mlflow.genai.evaluate()` with `Guidelines` and `Safety` scorers to assess summary quality

### Apache Spark
- **Distributed Preprocessing** -- `mapInPandas` with 16 workers for parallel MP3 → WAV conversion and chunking
- **Distributed Transcription** -- `mapInPandas` with 8 workers for parallel endpoint calls
- **Binary File Reader** -- `spark.read.format("binaryFile")` for discovering audio files at scale

### Delta Lake
- **MERGE INTO** -- Idempotent upserts to the transcription table (no duplicates on re-runs)
- **Liquid Clustering** -- `CLUSTER BY (parent_folder, filename)` for optimized query performance

### Databricks SDK
- **WorkspaceClient** -- Programmatic creation and management of the serving endpoint

## Cluster Requirements

You need **two clusters** -- one GPU cluster for model registration and one CPU cluster for the rest of the pipeline.

### GPU Cluster (for notebook `00_register_model`)

| Setting | Value |
|---------|-------|
| **Access Mode** | Single User |
| **Runtime** | ML Runtime 15.4 LTS GPU or later |
| **Node Type** | GPU instance with ≥16 GB GPU memory (e.g. `g5.2xlarge` on AWS, `Standard_NC6s_v3` on Azure) |
| **Workers** | 0 (single-node / driver-only) |
| **Use case** | Downloads the NVIDIA Parakeet TDT 0.6B model, runs a test inference, and logs it to Unity Catalog via MLflow |

> This cluster is only needed once for model registration. You can terminate it after `00_register_model` completes.

### CPU Cluster (for all other notebooks)

| Setting | Value |
|---------|-------|
| **Access Mode** | Single User |
| **Runtime** | ML Runtime 15.4 LTS or later (non-GPU) |
| **Node Type** | General purpose (e.g. `i3.xlarge` on AWS, `Standard_DS3_v2` on Azure) |
| **Workers** | 2–4 (for Spark parallelism in preprocessing and transcription) |
| **Use case** | Data download, audio preprocessing, transcription via endpoint, Delta merges, AI analysis, and evaluation |

### Notebook → Cluster mapping

| Notebook | Cluster |
|----------|---------|
| `00_register_model.py` | **GPU** |
| `00a_download_sample_dataset.py` | CPU |
| `00b_create_serving_endpoint.py` | CPU |
| `01_preprocess_audio_sparkparallism.py` | CPU |
| `02_transcribe_audio.py` | CPU |
| `03_merge_to_table.py` | CPU |
| `04_ai_analysis.py` | CPU |

### Additional Requirements

- Databricks workspace with Unity Catalog enabled
- Python packages (installed automatically via `%pip install` in each notebook): `nemo_toolkit[asr]`, `pydub`, `mlflow>=3.1.0`, `datasets`

## Acknowledgments

The sample audio dataset used in this project is provided by [Appen](https://huggingface.co/datasets/Appenlimited/1000h-us-english-smartphone-conversation):

> "USE-ASR003 Dataset Sample, Appen Butler Hill Pty Ltd, 2018."
