# EgoClip Dataset

## Prerequisites
The following requirements must be met to run the prepare_egoclip_dataset.py script, which generates the Ego4D VQA dataset:
1. Ego4D access, request it [here](https://ego4d-data.org/docs/start-here/). The files ego4d.json and nlq_train.json are required locally, as are the AWS credentials for access to the videos.
2. EgoClip metadata, stored in file egoclip.json, download it here.

## How to run
Perform the following steps, executing from the egoclip directory:
```python
python ./prepare_egoclip_dataset.py \
--ego4d_videos_path [relative path to ego4d.json] \
--egoclip_metadata [relative path to egoclip.json] \
--ego4d_trimmed_videos_path [path of trimmed videos] \
--egoclip_dataset [path to output JSON] \
--ego4d_aws_access_key_id [EGO4D_AWS_ACCESS_KEY_ID] \
--ego4d_aws_secret_access_key [EGO4D_AWS_SECRET_ACCESS_KEY] \
--ego4d_aws_region_name [EGO4D_AWS_REGION_NAME]
```
