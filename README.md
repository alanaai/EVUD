<div align=center>
<h2 align="center"> <a href="https://arxiv.org/abs/2406.13807">Egocentric Video Understanding Dataset (EVUD)</a></h2>

<img src="figures/EVUD_diagram.jpg" width="400px">

[![arXiv](https://img.shields.io/badge/arXiv-2046.13807-b31b1b.svg)](https://arxiv.org/abs/2046-13807)

<h5 align="center"> If you like our project, please give us a star :star: on GitHub for the latest update.</h5>

</div>

## TL;DR
We introduce the Egocentric Video Understanding Dataset (EVUD), an instruction-tuning dataset for training VLMs on video captioning and question answering tasks specific to egocentric videos.

## News
- The AlanaVLM paper is now on arXiv! [![arXiv](https://img.shields.io/badge/arXiv-2046.13807-b31b1b.svg)](https://arxiv.org/abs/2046-13807)
- Dataset and the code associated with our work will be released soon!

## Prerequisites

Create and activate virtual environment:
'''
python -m venv env
source venv/bin/activate
pip install -r ../requirements.txt
'''

## Data generation
Together with our generated data released on HuggingFace, we are also releasing all the scripts to reproduce our data generation pipeline:
- [Ego4D VQA](ego4d_vqa/README.md)
- [Ego4D VQA Gemini](gemini/README.md)
- [EgoClip](egoclip/README.md)
- [VSR](vsr/README.md)
- [HM3D](hm3d/README.md)

The generated data follows the format [LLaVa JSON format](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md).
