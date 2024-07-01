import json
import vertexai
from vertexai.generative_models import GenerativeModel
from tqdm import tqdm
import argparse
import time
import math
import os


################################################################################
# Instruction for prompting Gemini Pro
INSTRUCTION = """You are an intelligent embodied agent that can answer questions. You will be shown a video that was collected from a single location.

Your task is to generate a question for each of the following categories: object recognition, attribute recognition, object state recognition, object localisation, spatial reasoning, functional reasoning, world knowledge.

Ask diverse questions and give corresponding short answers. Include questions asking about the visual content of the video. The questions you posed can include the actions and behaviors of people or objects in the video, the chronological order of events, and causal relationships. Only include questions that have definite answers. Do not ask any questions that cannot be answered confidently.

Don't use headers. You should use the following format for each category:
Category: <category>
Question: <question>
Short answer: <answer>

Assistant:
"""


################################################################################
# Parse arguments
parser = argparse.ArgumentParser(
    description="Prompt Gemini Pro 1.5 to generate egocentric video understanding training data"
)
parser.add_argument(
    "--gcs_project_id", type=str, required=True, help="Your Google Cloud project ID"
)
parser.add_argument(
    "--gcs_bucket_name", type=str, required=True, help="GCS bucket with Ego4D NLQ clips"
)
parser.add_argument(
    "--gcs_location", type=str, required=True, help="GCS location to use with VertexAI"
)
parser.add_argument("--resume", action="store_true", help="Resume from last clip")
parser.add_argument(
    "--ego4d_vqa_gemini_path",
    type=str,
    default="./ego4d_vqa_gemini.json",
    help="Path to ego4d_vqa_gemini.json. Default: ./ego4d_vqa_gemini.json",
)
parser.add_argument(
    "--output_path",
    type=str,
    default="gemini_responses.json",
    help="Output path for Gemini responses. Default: gemini_responses.json",
)
parser.add_argument(
    "--gemini_model",
    type=str,
    default="gemini-1.5-pro-001",
    help="Gemini Pro model. Default: gemini-1.5-pro-001",
)
parser.add_argument(
    "--vertexai_quota",
    type=int,
    default=5,
    help="VertexAI request quota per minute. Default: 5",
)

args = parser.parse_args()
GCS_PROJECT_ID = args.gcs_project_id
GCS_BUCKET_NAME = args.gcs_bucket_name
GCS_LOCATION = args.gcs_location
RESUME = args.resume
NLQ_VQA_PATH = args.ego4d_vqa_gemini_path
OUTPUT_PATH = args.output_path
GEMINI_MODEL = args.gemini_model
QUOTA = args.vertexai_quota


################################################################################
# Load the NLQ VQA data
with open(NLQ_VQA_PATH, "r") as file:
    vqa = json.load(file)


################################################################################
# Initialize Vertex AI
vertexai.init(project=GCS_PROJECT_ID, location=GCS_LOCATION)

# Load the model
gemini = GenerativeModel(GEMINI_MODEL)


################################################################################
# Resume progress
if RESUME and os.path.exists(OUTPUT_PATH):
    with open(OUTPUT_PATH, "r") as file:
        responses = json.load(file)

else:
    responses = []

processed_clips = [r["example"]["id"] for r in responses]

if RESUME:
    print("-----------------------------------------------------------------------")
    print(f"Skipping {len(processed_clips)} already processed!")
    print("-----------------------------------------------------------------------")


################################################################################
# Process examples
time_limit = 60
time_queue = []

for idx, example in enumerate(tqdm(vqa, total=len(vqa))):
    if example["id"] in processed_clips:
        continue

    # record start time
    start_time = time.time()

    # Limit the requests to the specified quota
    if len(time_queue) == QUOTA:
        total_time = sum(time_queue)
        sleep_time = math.ceil(time_limit - total_time)
        if sleep_time > 0:
            time.sleep(sleep_time)
        time_queue.pop(0)

    clip_path = f"gs://{GCS_BUCKET_NAME}/{example['video_filename']}"
    clip = vertexai.generative_models.Part.from_uri(
        uri=clip_path, mime_type="video/mp4"
    )

    try:
        response = gemini.generate_content([clip, INSTRUCTION])
    except Exception:
        time.sleep(10)
        response = gemini.generate_content([clip, INSTRUCTION])

    responses.append(
        {
            "example": example,
            "response": response.to_dict(),
        }
    )

    # Output results every five responses
    if (idx + 1) % 5 == 0:
        # store results to JSON file
        with open(OUTPUT_PATH, "w") as out_file:
            json.dump(responses, out_file)

    # append duration to time queue to keep requests within specified quota
    time_queue.append(math.ceil(time.time() - start_time))

# store the last results to JSON file
with open(OUTPUT_PATH, "w") as out_file:
    json.dump(responses, out_file)

print("Done!")
