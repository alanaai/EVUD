import math
import random
import json
import copy
import argparse

random.seed(42)


################################################################################
# Utility/processing functions
def sort_dicts_by_category_list(dict_list, category_list):
    """Sorts a list of dictionaries based on the order of categories in a category list."""
    category_order = {category: i for i, category in enumerate(category_list)}
    sorted_dicts = sorted(
        dict_list, key=lambda item: category_order.get(item["category"], float("inf"))
    )
    return sorted_dicts


def preprocess_data(gen_data):
    """Process the generated text - check how many responses fit the expected format and return the missing data too"""
    categories = [
        "object recognition",
        "attribute recognition",
        "object state recognition",
        "object localisation",
        "spatial reasoning",
        "functional reasoning",
        "world knowledge",
    ]

    processed_data = []
    missing = []

    for g in gen_data:
        try:
            if len(g["response"]["candidates"]) < 1:
                missing.append(g)
                continue
            if "content" not in g["response"]["candidates"][0]:
                missing.append(g)
                continue

            response = g["response"]["candidates"][0]["content"]["parts"][0]["text"]
            category_data = response.split("\n\n")
            examples = []

            for c_data in category_data:
                c_lines = c_data.splitlines()
                # drop empty lines
                c_lines = [c_line for c_line in c_lines if bool(c_line)]
                example = {}
                for c_line in c_lines:
                    key, value = c_line.split(": ", 1)
                    example[key.strip().lower()] = (
                        value.strip().lower().replace("localization", "localisation")
                    )
                examples.append(example)

            # set the order of the examples to categories
            sorted_examples = sort_dicts_by_category_list(examples, categories)

            for example, category in zip(sorted_examples, categories):
                assert category == example["category"].lower()

            pd = copy.deepcopy(g)
            pd["processed_examples"] = sorted_examples
            processed_data.append(pd)

        except Exception:
            print(g)
            raise

    if len(missing) > 0:
        print(f"There are {len(missing)} examples which do not have a Gemini response")

    return processed_data, missing


def prepare_durations(train, missing_data, video_uid2video):
    """Get the durations, indexed by clip ID"""
    missing_keys = [
        f"{m['example']['video_uid']}-{m['example']['clip_uid']}-{m['example']['annotation_uid']}-{m['example']['language_query_index']}"
        for m in missing_data
    ]

    durations = {}
    for video in train["videos"]:
        for clip in video["clips"]:
            for annotation in clip["annotations"]:
                for language_query_index, language_query in enumerate(
                    annotation["language_queries"]
                ):
                    # check the video isn't in the missing set
                    key = f"{video['video_uid']}-{clip['clip_uid']}-{annotation['annotation_uid']}-{language_query_index}"
                    if key in missing_keys:
                        continue
                    start = max(math.floor(language_query["video_start_sec"]), 0)
                    end = min(
                        math.ceil(language_query["video_end_sec"]),
                        video_uid2video[video["video_uid"]]["duration_sec"],
                    )
                    duration = end - start
                    durations[key] = duration
    return durations


def prepare_turns(example):
    """Prepare a single example - shuffling categories randomly"""
    human_values = []
    gpt_values = []

    for pe in example["processed_examples"]:
        human_values.append({"from": "human", "value": pe["question"].capitalize()})
        gpt_values.append({"from": "gpt", "value": pe["short answer"].capitalize()})

    random_order = [0, 1, 2, 3, 4, 5, 6]
    random.shuffle(random_order)

    conversation = []
    for i, ro in enumerate(random_order):
        if i == 0:
            human_value = human_values[ro]
            human_value["value"] = f"<video>\n{human_value['value']}"
            conversation.append(human_value)
            conversation.append(gpt_values[ro])
        else:
            conversation.append(human_values[ro])
            conversation.append(gpt_values[ro])

    return conversation


def process_data(processed_data, durations):
    """Prepare all the examples into turned based dialogue that can be used for training"""
    dataset = []

    for d in processed_data:
        key = f"{d['example']['video_uid']}-{d['example']['clip_uid']}-{d['example']['annotation_uid']}-{d['example']['language_query_index']}"
        duration = durations[key]
        video_filename = d["example"]["video_filename"].replace(
            "ego4d_vqa_videos", "ego4d_vqa_gen_videos"
        )

        data_point = {
            "id": d["example"]["id"],
            "video": video_filename,
            "ego4d_video_uid": d["example"]["video_uid"],
            "ego4d_clip_uid": d["example"]["clip_uid"],
            "ego4d_annotation_uid": d["example"]["annotation_uid"],
            "ego4d_language_query_index": d["example"]["language_query_index"],
            "duration_(s)": duration,
            "human_annotated": False,
            "conversations": prepare_turns(d),
        }

        data_point["category_question_answer_tuples"] = d["processed_examples"]

        dataset.append(data_point)

    # shuffle the dataset
    random.shuffle(dataset)

    # now reindex the ids
    for idx, d in enumerate(dataset):
        d["id"] = idx

    return dataset


################################################################################
# Parse arguments
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--ego4d_path",
    type=str,
    default="../data/ego4d.json",
    help="Path to ego4d.json. Default: ../data/ego4d.json",
)
parser.add_argument(
    "--ego4d_nlq_path",
    type=str,
    default="../data/nlq_train.json",
    help="Path to nlq_train.json. Default: ../data/nlq_train.json",
)
parser.add_argument(
    "--gemini_data_path",
    type=str,
    default="gemini_responses.json",
    help="Path to Gemini responses. Default: gemini_responses.json",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="ego4d_nlq_train.gemini_pro_1.5.json",
    help="Output path for processed data. Default: ego4d_nlq_train.gemini_pro_1.5.json",
)

args = parser.parse_args()
EGO4D_META_PATH = args.ego4d_path
NLQ_TRAIN_PATH = args.ego4d_nlq_path
GEN_DATA_PATH = args.gemini_data_path
OUTPUT_PATH = args.output_path


################################################################################
# Load the required data
with open(EGO4D_META_PATH, "r") as file:
    meta = json.load(file)
video_uid2video = {video["video_uid"]: video for video in meta["videos"]}

with open(NLQ_TRAIN_PATH, "r") as file:
    train = json.load(file)

with open(GEN_DATA_PATH, "r") as file:
    gen_data = json.load(file)


################################################################################
# Process data
processed_data, missing_data = preprocess_data(gen_data)
durations = prepare_durations(train, missing_data, video_uid2video)
dataset = process_data(processed_data, durations)

# Dump to JSON file
with open(OUTPUT_PATH, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Outputted {len(dataset)} to {OUTPUT_PATH}")
print("Done!")
