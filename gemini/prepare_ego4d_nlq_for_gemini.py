import argparse
import json
import math
import os

import boto3
from moviepy.editor import VideoFileClip
from tqdm import tqdm
from google.cloud import storage


################################################################################
# GCS utility function
def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name, timeout=180)
    except Exception as e:
        print(
            f"Failed to upload {source_file_name} to {bucket_name}/{destination_blob_name}: {e}"
        )
        raise


################################################################################
# Parse arguments
parser = argparse.ArgumentParser(
    description="Process Ego4D NLQ Train clips for use with Gemini Pro"
)
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
    "--ego4d_output_videos_path",
    type=str,
    default="ego4d_vqa_gemini_videos/",
    help="Output video object path on GCS (and local path). Default: ego4d_vqa_gemini_videos",
)
parser.add_argument(
    "--output_json_path",
    type=str,
    default="ego4d_vqa_gemini.json",
    help="Path to output JSON file",
)
parser.add_argument(
    "--ego4d_aws_access_key_id",
    type=str,
    required=True,
    help="Ego4D AWS access key ID, obtained from Ego4D",
)
parser.add_argument(
    "--ego4d_aws_secret_access_key",
    type=str,
    required=True,
    help="Ego4D AWS secret access key, obtained from Ego4D",
)
parser.add_argument(
    "--ego4d_aws_region_name",
    type=str,
    required=True,
    help="Ego4D AWS region name, obtained from Ego4D",
)
parser.add_argument(
    "--gcs_bucket_name",
    type=str,
    required=True,
    help="GCS bucket the clips will be saved to",
)
parser.add_argument(
    "--keep-local-clips",
    action="store_true",
    help="Optional flag to specify keeping the clips locally (requires about 130 Gb of storage)",
)
args = parser.parse_args()


################################################################################
# Load the data
with open(args.ego4d_path) as in_file:
    ego4d_videos = json.load(in_file)
    video_uid2video = {video["video_uid"]: video for video in ego4d_videos["videos"]}

with open(args.ego4d_nlq_path) as in_file:
    ego4d_nlq = json.load(in_file)


################################################################################
# Process videos
dataset = []

# init JSON file
with open(args.output_json_path, "w") as in_file:
    pass

s3 = boto3.client(
    "s3",
    aws_access_key_id=args.ego4d_aws_access_key_id,
    aws_secret_access_key=args.ego4d_aws_secret_access_key,
    region_name=args.ego4d_aws_region_name,
)

last_downloaded_video_filename = None

for video in tqdm(ego4d_nlq["videos"], total=len(ego4d_nlq["videos"])):
    for clip in video["clips"]:
        for annotation in clip["annotations"]:
            for language_query_index, language_query in enumerate(
                annotation["language_queries"]
            ):
                idx = len(dataset)
                try:
                    s3_video_path_parts = video_uid2video[video["video_uid"]][
                        "s3_path"
                    ].split("/")
                    s3_bucket_name = s3_video_path_parts[2]
                    s3_key = "/".join(s3_video_path_parts[3:])
                    s3_filename = s3_video_path_parts[-3]

                    video_filename = os.path.join(
                        args.ego4d_output_videos_path,
                        video["video_uid"],
                    )

                    if video_filename != last_downloaded_video_filename:
                        if last_downloaded_video_filename:
                            os.remove(last_downloaded_video_filename)
                        s3.download_file(s3_bucket_name, s3_key, video_filename)
                        last_downloaded_video_filename = video_filename

                    video_start_sec = max(
                        math.floor(language_query["video_start_sec"]),
                        0,
                    )
                    video_end_sec = min(
                        math.ceil(language_query["video_end_sec"]),
                        video_uid2video[video["video_uid"]]["duration_sec"],
                    )

                    clip_filename = os.path.join(
                        args.ego4d_output_videos_path,
                        video["video_uid"],
                        clip["clip_uid"],
                        annotation["annotation_uid"],
                        f"{language_query_index}.mp4",
                    )

                    video_clip = VideoFileClip(last_downloaded_video_filename)
                    video_clip = video_clip.subclip(video_start_sec, video_end_sec)
                    video_clip.write_videofile(
                        clip_filename, remove_temp=True, logger=None
                    )

                    # Upload to GCS
                    if args.gcs_bucket_name:
                        upload_blob(args.gcs_bucket_name, clip_filename, clip_filename)

                    human_value = (f"<video>\n",)
                    if "query" in language_query:
                        human_value = f"<video>\n{language_query['query']}"

                    gpt_value = ""
                    if "answer" in language_query:
                        gpt_value = language_query["answer"].replace(
                            "Answer (Optional):", ""
                        )

                    dataset.append(
                        {
                            "id": idx,
                            "video_uid": video["video_uid"],
                            "clip_uid": clip["clip_uid"],
                            "annotation_uid": annotation["annotation_uid"],
                            "language_query_index": language_query_index,
                            "video_filename": clip_filename,
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": human_value,
                                },
                                {
                                    "from": "gpt",
                                    "value": gpt_value,
                                },
                            ],
                        }
                    )

                    with open(args.output_json_path, "w") as out_file:
                        json.dump(dataset, out_file)

                    if not args.keep_local_clips:
                        os.remove(clip_filename)

                except Exception as e:
                    print(f"Error with {idx}!")
                    print(e)

if os.path.exists(last_downloaded_video_filename):
    os.remove(last_downloaded_video_filename)

with open(args.output_json_path, "w") as out_file:
    json.dump(dataset, out_file)

print("Done!")
