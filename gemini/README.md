# Gemini-generated data

### Overview

You'll find three scripts in this directory for creating egocentric video understanding data using Gemini:
1. `prepare_ego4d_nlq_for_gemini.py`: prepare the Ego4D NLQ video clips for Gemini prompting.
2. `generate_gemini_data.py`: zero-shot multimodal prompting of Gemini to generate the training data. We used version `gemini-1.5-pro-preview-0409` for the published dataset, but we've updated the default to `gemini-1.5-pro-001`.
3. `prepare_ego4d_vqa_gemini_dataset.py`: post-processing Gemini output to prepare for training.

### Prerequisites

The following are required to run the above scripts:
1. Ego4D access, request it [here](https://ego4d-data.org/docs/start-here/). The files ego4d.json and nlq_train.json are required locally, as are the AWS credentials for access to the videos.
2. An [VertexAI](https://cloud.google.com/vertex-ai) API key for prompting Gemini.
3. A [GCS bucket](https://cloud.google.com/storage) for storing output Ego4D NLQ clips used for prompting Gemini.

### How to run

Perform the following steps, executing from the `gemini` directory:
``` python
# Create and activate virtual environment if you haven't already
python -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt

# Prepare the Ego4D NLQ data, the video clips will be uploaded to GCS, ready for prompting with VertexAI
python ./prepare_ego4d_nlq_for_gemini.py \
  --ego4d_path [relative path to ego4d.json] \                  # Default: ../data/ego4d.json
  --ego4d_nlq_path [relative path to nlq_train.json] \          # Default: ../data/nlq_train.json
  --ego4d_output_videos_path [path to output clips] \           # Output video object path on GCS (and local path). Default: ego4d_vqa_gemini_videos.
  --output_json_path [path to output JSON file] \               # Default: ego4d_vqa_gemini.json
  --ego4d_aws_access_key_id [EGO4D_AWS_ACCESS_KEY_ID] \         # Required, obtained from Ego4D
  --ego4d_aws_secret_access_key [EGO4D_AWS_SECRET_ACCESS_KEY] \ # Required, obtained from Ego4D
  --ego4d_aws_region_name [EGO4D_AWS_REGION_NAME] \             # Required, obtained from Ego4D
  --gcs_bucket_name [GCS_BUCKET_NAME] \                         # Required, GCS bucket the clips will be saved to
  --keep-local-clips                                            # Optional flag to specify keeping the clips locally (requires about 130 Gb of storage)

# Call VertexAI to generate training data
python ./generate_gemini_data.py \
  --gcs_project_id [GCS_PROJECT_ID] \                           # Required, your Google Cloud project ID
  --gcs_bucket_name [GCS_BUCKET_NAME] \                         # Required, GCS bucket with Ego4D NLQ clips
  --gcs_location [GCS_LOCATION] \                               # Required, GCS location to use with VertexAI
  --resume \                                                    # Optional flag to specify resuming from last clip
  --ego4d_vqa_gemini_path [path to Ego4D clips JSON file] \     # Outputted from previous script. Default: ./ego4d_vqa_gemini.json
  --output_path [path to output JSON file] \                    # Default: gemini_responses.json
  --gemini_model GEMINI_MODEL \                                 # Default: gemini-1.5-pro-001
  --vertexai_quota VERTEXAI_QUOTA                               # VertexAI request quota per minute. Default: 5

# Post-process the Gemini data to create JSON used for training
python ./prepare_ego4d_vqa_gemini_dataset.py \
  --ego4d_path [path to ego4d.json] \                           # Default: ../data/ego4d.json
  --ego4d_nlq_path [path to nlq_train.json] \                   # Default: ../data/nlq_train.json
  --gemini_data_path [path to Gemini responses JSON file] \     # Outputted from previous script. Default: gemini_responses.json
  --output_path [path to output JSON file]                      # Default: ../output/ft_json/gemini.json
```

#### Note

After performing human annotation, we manually replaced the Gemini-generated answers with the gold standard answers for inclusion in the EVUD dataset.