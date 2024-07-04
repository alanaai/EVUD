import argparse
import json
import os
import random
from collections import defaultdict

import ollama
from datasets import load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vsr_questions",
        type=str,
        default="../output/ft_json/vsr_questions.json",
    )
    args = parser.parse_args()

    random.seed(42)

    data_files = {
        "train": "train.jsonl",
        "validation": "dev.jsonl",
        "test": "test.jsonl",
    }
    dataset = load_dataset("cambridgeltl/vsr_random", data_files=data_files)
    image2instances = defaultdict(list)

    for instance in tqdm(dataset["train"], total=len(dataset["train"])):
        image = instance["image"]
        caption = instance["caption"]
        label = instance["label"]
        label = ["True", "Yes"] if label else ["False", "No"]
        answer = random.choice(label)

        response = ollama.chat(
            model="llama3",
            messages=[
                {
                    "role": "user",
                    "content": f"""
            Generate a polar question from the following statement about a picture.
            Keep as many words as you can of the statement in the question and do not add unnecessary words.
            Always generate just the question.
            Do not include any explanations.
            Statement: {caption}
            """,
                },
            ],
        )
        question = f"{response['message']['content']}"

        image2instances[image].append((question, answer))

    dataset = []
    for image, instances in image2instances.items():
        conversations = []
        for question, answer in instances:
            conversations.append(
                {
                    "from": "human",
                    "value": f"<image>\n{question}",
                },
            )
            conversations.append(
                {"from": "gpt", "value": answer},
            )

        dataset.append(
            {
                "id": len(dataset),
                "image": os.path.join("vsr_images", image),
                "conversations": conversations,
            }
        )

    with open(args.vsr_questions, "w") as out_file:
        json.dump(dataset, out_file)
