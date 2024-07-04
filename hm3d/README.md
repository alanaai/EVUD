# HM3D Dataset

## Prerequisites
**Due to the Habitat-Sim requirements, you must run this script on a Linux GPU-enabled machine.**

The following requirements must be met to run the prepare_hm3d_dataset.py script, which generates the Ego4D VQA dataset:
1. Before running this script, create a Conda environment following the instructions listed [here](https://github.com/facebookresearch/habitat-sim/blob/main/README.md#recommended-conda-packages).
2. OpenEQA dataset, download it here, stored in the file [open-eqa-v0.json](https://github.com/facebookresearch/open-eqa/blob/main/data/open-eqa-v0.json).
3. HM3D, which can be downloaded following the [Habitat-Sim documentation](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d).

## How to run
Perform the following steps, executing from the hm3d directory:
```python
python ./prepare_hm3d_dataset.py \
--openeqa_dataset [relative path to open-eqa-v0.json] \
--egoclip_metadata [relative path to the path of Matterport data] \
--video_output_dir [path of generated videos] \
--annotations_filename [path to output JSON]
```
