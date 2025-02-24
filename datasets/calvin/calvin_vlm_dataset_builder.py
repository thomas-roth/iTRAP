import datetime
import os
import sys
import logging
import json
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).absolute().parents[2]))
from datasets.calvin.calvin_dataset_builder import CalvinDatasetBuilder


class CalvinVLMDatasetBuilder(CalvinDatasetBuilder):
    def __init__(self, timestamp, dataset_path="/DATA/calvin/task_D_D", traj_simplification_rdp_epsilon=0.01,
                 output_dir="/home/troth/bt/data/calvin_vlm_dataset", traj_string_coords_precision=3):
        
        self.timestamp = timestamp
        super().__init__(dataset_path, traj_simplification_rdp_epsilon)
        self.output_dir = output_dir
        self.traj_string_coords_precision = traj_string_coords_precision # images are 200x200 & 84x84 => 1 pixel >= 0.005

        os.makedirs(f"{self.output_dir}/{self.timestamp}", exist_ok=True)

        _file_handler = logging.FileHandler(f"{self.output_dir}/{self.timestamp}/build_dataset.log", mode='w')
        _file_handler.setLevel(logging.INFO)
        _file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
        self._logger.addHandler(_file_handler)

        self._logger.info("Initialized CalvinVLMDatasetBuilder")


    def build_trajectory_representation(self, gripper_centers_world, gripper_widths):
        # project simplified trajectory to static image space & generate trajectory string

        gripper_centers = self._project_gripper_centers_to_cam(gripper_centers_world, cam_id=0)

        assert self.env.cameras[0].width == self.env.cameras[0].height
        assert self.env.cameras[1].width == self.env.cameras[1].height

        is_gripper_open = gripper_widths[0] == self.CALVIN_GRIPPER_WIDTH_OPEN
        traj_string_contents = []
        for (gripper_center, gripper_width) in zip(gripper_centers, gripper_widths):
            normalized_gripper_center_x = round(float(gripper_center[0]) / self.env.cameras[0].width, self.traj_string_coords_precision)
            normalized_gripper_center_y = round(float(gripper_center[1]) / self.env.cameras[1].width, self.traj_string_coords_precision)

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


    def _save_trajectory_strings(self, task_all_seqs, traj_strings_all_seqs, start_imgs_all_seqs, dataset_split):
        assert len(task_all_seqs) == len(traj_strings_all_seqs) == len(start_imgs_all_seqs)

        num_digits = len(str(len(task_all_seqs)))

        dataset_path = f"{self.output_dir}/{self.timestamp}/{dataset_split}"
        os.makedirs(dataset_path, exist_ok=True)

        dataset_entries = []
        for i, (task_seq, traj_string_seq, start_imgs_seq) in enumerate(zip(task_all_seqs, traj_strings_all_seqs, start_imgs_all_seqs)):
            first_static_img = Image.fromarray(start_imgs_seq["rgb_static"])
            first_static_img_path = f"{dataset_path}/{i:0{num_digits}d}_{task_seq}_static.png"
            first_static_img.save(first_static_img_path)

            first_gripper_img = Image.fromarray(start_imgs_seq["rgb_gripper"])
            first_gripper_img_path = f"{dataset_path}/{i:0{num_digits}d}_{task_seq}_gripper.png"
            first_gripper_img.save(first_gripper_img_path)

            # FIXME?: use long version of task instruction for prompt (["language"]["ann"] instead of ["language"]["task"])
            dataset_entry = {
                "messages": [{
                    "content": f"<image><image>In the images, please execute the command described in <prompt>{str.replace(task_seq, '_', ' ')}</prompt>. " \
                                "Provide a sequence of points denoting the trajectory of a robot gripper to achieve the goal. " \
                                "Format your answer as a list of tuples enclosed by <ans> and </ans> tags. For example: <ans>[(0.252, 0.328), (0.327, 0.174), " \
                                "(0.139, 0.242), <action>Open Gripper</action>, (0.746, 0.218), <action>Close Gripper</action>, ...]</ans>. Each tuple denotes " \
                                "an x and y location of the end effector of the gripper in the first image. The action tags indicate the gripper action. " \
                                "The coordinates should be floats ranging between 0 and 1, indicating the relative locations of the points in the first image.",
                    "role": "user"
                },{
                    "content": traj_string_seq,
                    "role": "assistant"
                }],
                "images": [
                    first_static_img_path,
                    first_gripper_img_path
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


    def build_dataset(self):
        for dataset_split in ["training", "validation"]:
            task_all_seqs, traj_strings_all_seqs, start_imgs_all_seqs, _ = self._build_trajectories(dataset_split)

            self._save_trajectory_strings(task_all_seqs, traj_strings_all_seqs, start_imgs_all_seqs, dataset_split)
        
        self._logger.info("Finished building VLM dataset")


if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    calvin_vlm_dataset_builder = CalvinVLMDatasetBuilder(timestamp=timestamp)
    calvin_vlm_dataset_builder.build_dataset()
