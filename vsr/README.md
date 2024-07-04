# VSR Dataset

## Prerequisites
Ollama is required to run the prepare_vsr_dataset.py script, which generates the VSR dataset. Moreover, the llama3 model must be pulled through Ollama.

The following requirements must be met to run the prepare_vsr_dataset.py script, which generates the VSR dataset:
1. Download [Ollama](https://ollama.com).
2. ollama pull llama3
3. ollama serve

## How to run
Perform the following steps, executing from the vsr directory:
``` python
python ./prepare_vsr_dataset.py \
--vsr_questions [path to output JSON]
```
