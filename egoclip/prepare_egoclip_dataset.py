import argparse
import json
import os
import random

import boto3
import pandas as pd
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def prepare_egoclip(egoclip_metadata, num_clips=50000, min_duration=2, max_duration=60):
    df = pd.read_csv(
        egoclip_metadata, sep="\t", on_bad_lines=lambda x: x[:-1], engine="python"
    )

    prompts = [
        "Can you please provide a brief description of the video?",
        "Describe the content of the video.",
        "What is happening in the video? Please describe it.",
        "Can you summarize the key events or actions in the video?",
        "Describe the visual elements and any notable features in the video.",
        "Provide a narrative description of the video.",
        "What's in the video?",
        "What can you see in this video?",
        "What's happening in the video?",
        "What is the main focus of the video?",
    ]

    def prompt_selection(row):
        prompt = random.choice(prompts)
        return prompt

    df = df.drop("Unnamed: 10", axis=1)
    filtered_df = df[
        (df["clip_end"] - df["clip_start"] >= min_duration)
        & (df["clip_end"] - df["clip_start"] <= max_duration)
    ]
    sub_df = filtered_df.sample(n=num_clips, random_state=42)

    sub_df["instruction"] = sub_df.apply(prompt_selection, axis=1)

    """
    There are four flags that annotators use in the sentence boxes:

    #unsure to denote they are unsure about a specific statement
    #summary to denote they are giving the overall video
    #C to denote the sentence is an action done by the camera wearer (the person who recorded the video while wearing a camera on their head)
    #O to denote that the sentence is an action done by someone other than the camera wearer

    Note that every sentence will have either #C or #O. 
    Only some sentences (or none) may have #unsure. 
    Only one sentence for the entire video clip will have #summary.
    """

    def transform_clip_text(row):
        clip_text = row["clip_text"]

        # This tag was used to denote that the annotator was unsure about a specific object/statement
        clip_text = clip_text.replace("#UNSURE", "something")

        if clip_text.startswith("Summary"):
            # Simply remove the summary tag if it exists
            clip_text = clip_text.replace("Summary", "")
        elif clip_text.startswith("#C"):
            clip_text = clip_text.replace("#C", "")
        elif clip_text.startswith("#O"):
            clip_text = clip_text.replace("#O", "")

        clip_text = clip_text.replace("C", "the camera wearer", 1).replace(
            "O", "another person", 1
        )

        clip_text = clip_text.strip()

        return (clip_text + ".").capitalize()

    sub_df["clip_text_refined"] = sub_df.apply(transform_clip_text, axis=1)

    return sub_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ego4d_videos_path",
        type=str,
        default="../data/ego4d.json",
    )
    parser.add_argument(
        "--egoclip_metadata",
        type=str,
        default="../data/egoclip.csv",
    )
    parser.add_argument(
        "--ego4d_trimmed_videos_path",
        type=str,
        default="../output/egoclip_videos/",
    )
    parser.add_argument(
        "--egoclip_dataset",
        type=str,
        default="../output/ft_json/egoclip.json",
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
    args = parser.parse_args()

    with open(args.ego4d_videos_path) as in_file:
        ego4d_videos = json.load(in_file)
        video_uid2video = {
            video["video_uid"]: video for video in ego4d_videos["videos"]
        }

    egoclip_metadata = prepare_egoclip(args.egoclip_metadata)

    egoclip_metadata = egoclip_metadata.sort_values("video_uid")

    dataset = []

    s3 = boto3.client(
        "s3",
        aws_access_key_id=args.ego4d_aws_access_key_id,
        aws_secret_access_key=args.ego4d_aws_secret_access_key,
        region_name=args.ego4d_aws_region_name,
    )

    last_downloaded_video_filename = None
    last_downloaded_video = None

    for index, row in tqdm(egoclip_metadata.iterrows(), total=len(egoclip_metadata)):
        video_uid = row["video_uid"]
        s3_video_path_parts = video_uid2video[video_uid]["s3_path"].split("/")
        s3_bucket_name = s3_video_path_parts[2]
        s3_key = "/".join(s3_video_path_parts[3:])
        s3_filename = s3_video_path_parts[-3]
        video_filename = video_uid + ".mp4"

        if video_filename != last_downloaded_video_filename:
            if last_downloaded_video_filename:
                os.remove(last_downloaded_video_filename)
                last_downloaded_video.close()
                del last_downloaded_video
            s3.download_file(s3_bucket_name, s3_key, video_filename)
            last_downloaded_video_filename = video_filename
            last_downloaded_video = VideoFileClip(last_downloaded_video_filename)

        video_start_sec = row["clip_start"]
        video_end_sec = min(row["clip_end"], last_downloaded_video.duration)
        trimmed_video_path = os.path.join(args.ego4d_trimmed_videos_path, video_uid)

        os.makedirs(trimmed_video_path, exist_ok=True)

        trimmed_video_filename = os.path.join(
            trimmed_video_path, f"{row['narration_ind']}.mp4"
        )

        try:
            video_clip = last_downloaded_video.subclip(video_start_sec, video_end_sec)
            video_clip.write_videofile(
                trimmed_video_filename, remove_temp=True, logger=None
            )

            print(f"Trimmed video saved to: {trimmed_video_filename}")

            dataset.append(
                {
                    "id": len(dataset),
                    "video": trimmed_video_filename,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<video>\n{row['instruction']}",
                        },
                        {"from": "gpt", "value": row["clip_text_refined"]},
                    ],
                }
            )
        except (OSError, Exception):
            print(f"Skipping {trimmed_video_filename}")
            continue

    if last_downloaded_video_filename is not None and os.path.exists(
        last_downloaded_video_filename
    ):
        os.remove(last_downloaded_video_filename)

    with open(args.egoclip_dataset, "w") as out_file:
        json.dump(dataset, out_file)
