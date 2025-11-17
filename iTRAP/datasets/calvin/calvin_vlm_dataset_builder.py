import argparse
import datetime
import math
import os
import logging
import json
from pathlib import Path
import sys
from PIL import Image
from tqdm import tqdm

from calvin_dataset_builder import CalvinDatasetBuilder

sys.path.append(str(Path(__file__).absolute().parents[3])) # Add repo root to path
from iTRAP.models.Qwen3_VL.resize_utils import resize_point_for_qwen3_vl
from iTRAP.models.Qwen3_VL.utils import get_prompt

# add calvin_env to path
sys.path.append(str(Path(__file__).absolute().parents[2] / "models" / "flower_vla_calvin" / "calvin_env"))


class CalvinVLMDatasetBuilder(CalvinDatasetBuilder):
    def __init__(self, timestamp, dataset_path, output_dir,
                 traj_simplification_rdp_epsilon=0.01, traj_string_coords_precision=3):
        
        self.timestamp = timestamp
        super().__init__(dataset_path, traj_simplification_rdp_epsilon)
        self.output_dir = output_dir
        self.traj_string_coords_precision = traj_string_coords_precision # images are 200x200 & 84x84 => 1 pixel >= 0.005

        os.makedirs(f"{self.output_dir}/{self.timestamp}", exist_ok=False)

        _file_handler = logging.FileHandler(f"{self.output_dir}/{self.timestamp}/build_dataset.log", mode='w')
        _file_handler.setLevel(logging.INFO)
        _file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        self._logger.addHandler(_file_handler)

        self._logger.info("Initialized CalvinVLMDatasetBuilder")


    def build_trajectory_representation(self, gripper_centers_world, gripper_widths):
        # project simplified trajectory to static image space & generate trajectory string

        gripper_centers = self._project_gripper_centers_to_cam(gripper_centers_world, cam_id=0)

        assert self.env.cameras[0].width == self.env.cameras[0].height
        static_img_size = self.env.cameras[0].width

        is_gripper_open = gripper_widths[0] == self.CALVIN_GRIPPER_WIDTH_OPEN
        traj_string_contents = []
        for (gripper_center, gripper_width) in zip(gripper_centers, gripper_widths):
            # resize gripper center to match image resizing of Qwen3-VL
            normalized_gripper_center_x, normalized_gripper_center_y = resize_point_for_qwen3_vl(gripper_center, static_img_size, static_img_size)

            traj_string_contents.append(f"({normalized_gripper_center_x}, {normalized_gripper_center_y})")

            if is_gripper_open and gripper_width == self.CALVIN_GRIPPER_WIDTH_CLOSED:
                traj_string_contents.append("<action>Close Gripper</action>")
                is_gripper_open = False
            elif not is_gripper_open and gripper_width == self.CALVIN_GRIPPER_WIDTH_OPEN:
                traj_string_contents.append("<action>Open Gripper</action>")
                is_gripper_open = True

        traj_string_seq = "<ans>[" + str.join(", ", traj_string_contents) + "]</ans>"
        start_imgs_seq = {
            "rgb_static": self.curr_seq["obs"]["rgb_static"][0],
            "rgb_gripper": self.curr_seq["obs"]["rgb_gripper"][0]
        }

        return {"traj_string_seq": traj_string_seq, "start_imgs_seq": start_imgs_seq}


    def _save_trajectory_strings(self, task_all_seqs, task_text_all_seqs, traj_strings_all_seqs, start_imgs_all_seqs, dataset_split):
        assert len(task_all_seqs) == len(task_text_all_seqs), f"len(task_all_seqs) ({len(task_all_seqs)}) != len(task_text_all_seqs) ({len(task_text_all_seqs)})"
        assert len(task_all_seqs) == len(traj_strings_all_seqs), f"len(task_all_seqs) ({len(task_all_seqs)}) != len(traj_strings_all_seqs) ({len(traj_strings_all_seqs)})"
        assert len(task_all_seqs) == len(start_imgs_all_seqs), f"len(task_all_seqs) ({len(task_all_seqs)}) != len(start_imgs_all_seqs) ({len(start_imgs_all_seqs)})"

        num_digits = len(str(len(task_all_seqs)))

        dataset_path = f"{self.output_dir}/{self.timestamp}/{dataset_split}"
        os.makedirs(dataset_path, exist_ok=True)

        dataset_entries = []
        for i, (task_seq, task_text_seq, traj_string_seq, start_imgs_seq) in tqdm(enumerate(zip(task_all_seqs, task_text_all_seqs, traj_strings_all_seqs, start_imgs_all_seqs)),
                                                                                  total=len(task_all_seqs), desc=f"Building question-answer pairs for {dataset_split} split"):
            first_static_img = Image.fromarray(start_imgs_seq["rgb_static"])
            first_static_img_name = f"{i:0{num_digits}d}_{task_seq}_static.png"
            first_static_img.save(f"{dataset_path}/{first_static_img_name}")

            first_gripper_img = Image.fromarray(start_imgs_seq["rgb_gripper"])
            first_gripper_img_name = f"{i:0{num_digits}d}_{task_seq}_gripper.png"
            first_gripper_img.save(f"{dataset_path}/{first_gripper_img_name}")

            prompt = get_prompt(task_text_seq)

            dataset_entry = {
                "messages": [{
                    "content": prompt,
                    "role": "user"
                },{
                    "content": traj_string_seq,
                    "role": "assistant"
                }],
                "images": [
                    first_static_img_name,
                    first_gripper_img_name
                ]
            }
            dataset_entries.append(dataset_entry)

        dataset_info = {
            "dataset": {
                "file_name": "dataset.json",
                "formatting": "sharegpt",
                "columns": {
                    "messages": "messages",
                    "images": "images"
                    },
                "tags": {
                    "role_tag": "role",
                    "content_tag": "content",
                    "user_tag": "user",
                    "assistant_tag": "assistant"
                    }
            }
        }

        with open(f"{dataset_path}/dataset.json", "w") as dataset_file:
            json.dump(dataset_entries, dataset_file, indent=2)
        with open(f"{dataset_path}/dataset_info.json", "w") as dataset_info_file:
            dataset_info_str = json.dumps(dataset_info, indent=2)[1:-1].strip() # remove outer curly braces
            dataset_info_file.write(dataset_info_str)
        
        self._logger.info(f"Built dataset file for {dataset_split} split")


    def build_dataset(self):
        for dataset_split in ["training", "validation"]:
            task_all_seqs, task_text_all_seqs, traj_strings_all_seqs, start_imgs_all_seqs, _ = self._build_trajectories(dataset_split)

            self._save_trajectory_strings(task_all_seqs, task_text_all_seqs, traj_strings_all_seqs, start_imgs_all_seqs, dataset_split)
        
        self._logger.info("Finished building VLM dataset")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="/DATA/calvin/task_ABC_D")
    parser.add_argument("--output-dir", type=str, default="/home/troth/data/iTRAP-flower/calvin_vlm_dataset")
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    calvin_vlm_dataset_builder = CalvinVLMDatasetBuilder(timestamp=timestamp, dataset_path=args.dataset_path, output_dir=args.output_dir)
    calvin_vlm_dataset_builder.build_dataset()
