# Ego4D VQA Dataset

## Prerequisites
Ego4D access, request it [here](https://ego4d-data.org/docs/start-here/), is required to run the prepare_ego4d_vqa_dataset.py script, which generates the Ego4D VQA dataset. The files ego4d.json and nlq_train.json are required locally, as are the AWS credentials for access to the videos.

## How to run
Perform the following steps, executing from the ego4d_vqa directory:
```python
python ./prepare_ego4d_vqa_dataset.py \
--ego4d_videos_path [relative path to ego4d.json] \
--ego4d_nlq_path [relative path to nlq_train.json] \
--min_duration [minimum duration of videos] \
--max_duration [maximum duration of videos] \
--ego4d_trimmed_videos_path [path of trimmed videos] \
--ego4d_vqa_path [path to output JSON] \
--ego4d_aws_access_key_id [EGO4D_AWS_ACCESS_KEY_ID] \
--ego4d_aws_secret_access_key [EGO4D_AWS_SECRET_ACCESS_KEY] \
--ego4d_aws_region_name [EGO4D_AWS_REGION_NAME]
```
