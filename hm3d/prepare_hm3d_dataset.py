import argparse
import glob
import json
import multiprocessing
import os
import random
from collections import Counter
from copy import deepcopy
from functools import partial

import cv2
import habitat_sim
import numpy as np
import spacy
from habitat_sim.nav import ShortestPath
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from tqdm import tqdm


class HabitatDataGenerator:
    def __init__(self, scene_dataset_config_filename, scene_filename):
        self._scene_dataset_config_filename = scene_dataset_config_filename
        self._scene_filename = scene_filename

        self._init_simulator()

    def _init_simulator(self):
        settings = deepcopy(default_sim_settings)
        settings.update(
            {
                "max_frames": 1000,
                "scene_dataset_config_file": self._scene_dataset_config_filename,
                "scene": self._scene_filename,
                "color_sensor": True,
            }
        )
        self._settings = settings
        self._cfg = make_cfg(settings)

        self._sim = habitat_sim.Simulator(self._cfg)

        random.seed(42)
        self._sim.seed(42)

    def _init_agent_state(self, agent_id, goal_position, min_distance):
        # Initialize the agent at a random start state
        agent = self._sim.initialize_agent(agent_id)
        start_state = agent.get_state()

        num_start_tries = 0
        while (
            start_state.position[1] > 0.5
            or self._compute_shortest_path(
                start_state.position, goal_position
            ).geodesic_distance
            < min_distance
        ) and num_start_tries < 100:
            start_state.position = self._sim.pathfinder.get_random_navigable_point()
            num_start_tries += 1
        agent.set_state(start_state)

        if num_start_tries == 100:
            return None

        return start_state

    def _compute_shortest_path(self, start_pos, end_pos):
        shortest_path = ShortestPath()
        shortest_path.requested_start = start_pos
        shortest_path.requested_end = end_pos
        self._sim.pathfinder.find_path(shortest_path)
        return shortest_path

    def generate_video(self, goal_position, video_filename, min_distance=10):
        agent_id = self._settings["default_agent"]

        start_state = self._init_agent_state(agent_id, goal_position, min_distance)

        if start_state is None:
            return False

        greedy_follower = self._sim.make_greedy_follower(agent_id=agent_id)

        try:
            action_path = greedy_follower.find_path(goal_position)
        except habitat_sim.errors.GreedyFollowerError:
            return False

        if not action_path:
            return False

        # Assuming you have a list of preprocessed images (tensors) in 'images'
        writer = cv2.VideoWriter(
            video_filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            # (640, 480),
            (self._settings["width"], self._settings["height"]),
        )  # Adjust codec and frame size

        for action in action_path:
            if action is None:
                continue

            observation = self._sim.step(action)

            color_obs = observation["color_sensor"]

            frame = cv2.cvtColor(color_obs, cv2.COLOR_RGB2BGR)
            writer.write(frame)

        writer.release()

        return True

    def get_relevant_objects(self, openeqa_objects_counter: Counter, max_num_objects):
        scene_specific_objects = [
            x
            for x in self._sim.semantic_scene.objects
            if x.category.name() in openeqa_objects_counter
        ]

        if len(scene_specific_objects) <= max_num_objects:
            return scene_specific_objects

        # Otherwise we have to sample following the distributiion of the OpenEQA data
        total_mass = sum(
            openeqa_objects_counter[x.category.name()] for x in scene_specific_objects
        )

        weights = []
        values = []

        for obj in scene_specific_objects:
            values.append(obj)
            obj_freq = openeqa_objects_counter[obj.category.name()]
            weights.append(obj_freq / total_mass)

        return np.random.choice(values, size=max_num_objects, p=weights).tolist()

    def close(self):
        self._sim.close()


def generate_caption(category_name):
    templates = [
        "The person is walking towards the {}.",
        "The camera wearer is walking towards the {}.",
        "I'm walking over to the {} now.",
        "There's a person approaching the {}.",
        "I see a person approaching the {}.",
        "I see someone walking up to the {}.",
    ]

    return random.choice(templates).format(category_name)


def generate_videos_from_scene(args, openeqa_objects_counter: Counter, scene):
    generator = HabitatDataGenerator(scene[0], scene[1])

    relevant_objects = generator.get_relevant_objects(
        openeqa_objects_counter, args.max_num_objects
    )

    return_values = []
    scene_name = scene[1].split("/")[-1].replace(".glb", "")
    scene_dirname = os.path.join(args.video_output_dir, f"scene_{scene_name}")

    os.makedirs(scene_dirname, exist_ok=True)

    for object in relevant_objects:
        goal_position = object.aabb.center
        object_name = object.category.name()
        video_filename = os.path.join(scene_dirname, f"object_{object_name}.mp4")

        if generator.generate_video(goal_position, video_filename):
            caption = generate_caption(object.category.name())

            relative_video_filename = os.path.join(
                *video_filename.split(os.path.sep)[-3:]
            )
            return_values.append((relative_video_filename, caption))

    generator.close()

    return return_values


def find_scene_files():
    scenes = []
    scene_dataset_config_file = "hm3d_annotated_basis.scene_dataset_config.json"

    for file in glob.glob("train/**/*.glb"):
        if "semantic" not in file:
            semantic_file = file.replace(".glb", ".semantic.glb")

            if os.path.exists(semantic_file):
                scenes.append((scene_dataset_config_file, file))

    return scenes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--openeqa_dataset",
        type=str,
        default="../data/open-eqa-v0.json",
    )
    parser.add_argument(
        "--main_data_root",
        type=str,
        default="../matterport_data/scene_datasets/hm3d/",
    )
    parser.add_argument(
        "--video_output_dir", type=str, default="../output/hm3d_gen_videos/"
    )
    parser.add_argument(
        "--annotations_filename",
        type=str,
        default="../output/ft_json/hm3d_captions.json",
    )
    parser.add_argument("--num_workers", default=1, type=int)
    parser.add_argument("--max_num_objects", default=30, type=int)

    args = parser.parse_args()

    with open(args.openeqa_dataset) as in_file:
        openeqa_dataset = json.load(in_file)

    nlp = spacy.load("en_core_web_sm", disable=["parser"])

    openeqa_objects = []

    ignore_objects = {
        "end",
        "wall",
        "floor",
        "ceiling",
        "items",
        "c",
        "l",
        "left",
        "right",
        "center",
        "-",
        "room",
    }

    for qa in openeqa_dataset:
        answer = qa["answer"]

        for token in nlp(answer):
            if token.pos_ == "NOUN" and token.lower_ not in ignore_objects:
                openeqa_objects.append(token.lower_)

    openeqa_objects = Counter(openeqa_objects)
    print(f"# OpenEQA objects: {len(openeqa_objects)}")
    print(openeqa_objects.most_common(10))

    os.chdir(args.main_data_root)

    scenes = find_scene_files()

    print(f"Found a total of {len(scenes)} scenes in the `train` folder.")

    dataset = []

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

    with multiprocessing.Pool(args.num_workers) as pool:
        with tqdm(total=len(scenes)) as progress_bar:
            for idx, return_values in enumerate(
                pool.imap_unordered(
                    partial(generate_videos_from_scene, args, openeqa_objects), scenes
                )
            ):
                progress_bar.update(1)

                for video_filename, caption in return_values:
                    prompt = random.choice(prompts)

                    dataset.append(
                        {
                            "id": len(dataset),
                            "video": video_filename,
                            "conversations": [
                                {
                                    "from": "human",
                                    "value": f"<video>\n{prompt}",
                                },
                                {"from": "gpt", "value": caption},
                            ],
                        }
                    )

    with open(args.annotations_filename, "w") as out_file:
        json.dump(dataset, out_file)
