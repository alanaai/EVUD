import argparse
import json
import os

import boto3
from moviepy.editor import VideoFileClip
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ego4d_videos_path",
        type=str,
        default="../data/ego4d.json",
    )
    parser.add_argument(
        "--ego4d_nlq_path",
        type=str,
        default="../data/nlq_train.json",
    )
    parser.add_argument(
        "--min_duration",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max_duration",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--ego4d_trimmed_videos_path",
        type=str,
        default="../output/ego4d_vqa_videos/",
    )
    parser.add_argument(
        "--ego4d_vqa_path",
        type=str,
        default="../output/ft_json/ego4d_vqa.json",
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

    with open(args.ego4d_nlq_path) as in_file:
        ego4d_nlq = json.load(in_file)

    dataset = []

    s3 = boto3.client(
        "s3",
        aws_access_key_id=args.ego4d_aws_access_key_id,
        aws_secret_access_key=args.ego4d_aws_secret_access_key,
        region_name=args.ego4d_aws_region_name,
    )

    last_downloaded_video_filename = None
    last_downloaded_video = None

    for video in tqdm(ego4d_nlq["videos"], total=len(ego4d_nlq["videos"])):
        for clip in video["clips"]:
            for annotation in clip["annotations"]:
                for language_query_index, language_query in enumerate(
                    annotation["language_queries"]
                ):
                    if "answer" in language_query:
                        s3_video_path_parts = video_uid2video[video["video_uid"]][
                            "s3_path"
                        ].split("/")
                        s3_bucket_name = s3_video_path_parts[2]
                        s3_key = "/".join(s3_video_path_parts[3:])
                        s3_filename = s3_video_path_parts[-3]
                        video_filename = video["video_uid"]

                        if video_filename != last_downloaded_video_filename:
                            if last_downloaded_video_filename:
                                os.remove(last_downloaded_video_filename)
                                last_downloaded_video.close()
                                del last_downloaded_video
                            s3.download_file(s3_bucket_name, s3_key, video_filename)
                            last_downloaded_video_filename = video_filename
                            last_downloaded_video = VideoFileClip(
                                last_downloaded_video_filename
                            )

                        video_start_sec = max(
                            language_query["video_start_sec"],
                            0,
                        )
                        video_end_sec = min(
                            language_query["video_end_sec"],
                            video_uid2video[video["video_uid"]]["duration_sec"],
                        )

                        if (
                            args.min_duration
                            <= video_end_sec - video_start_sec
                            <= args.max_duration
                        ):
                            trimmed_video_path = os.path.join(
                                args.ego4d_trimmed_videos_path,
                                video["video_uid"],
                                clip["clip_uid"],
                                annotation["annotation_uid"],
                            )

                            os.makedirs(trimmed_video_path, exist_ok=True)

                            trimmed_video_filename = os.path.join(
                                trimmed_video_path, f"{language_query_index}.mp4"
                            )

                            video_clip = last_downloaded_video.subclip(
                                video_start_sec, video_end_sec
                            )
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
                                            "value": f"<video>\n{language_query['query']}",
                                        },
                                        {
                                            "from": "gpt",
                                            "value": language_query["answer"].replace(
                                                "Answer (Optional):", ""
                                            ),
                                        },
                                    ],
                                }
                            )

    if os.path.exists(last_downloaded_video_filename):
        os.remove(last_downloaded_video_filename)

    with open(args.ego4d_vqa_path, "w") as out_file:
        json.dump(dataset, out_file)
