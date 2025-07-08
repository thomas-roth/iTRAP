import argparse
import datetime
import os
import shutil
import sys
import logging
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from calvin_dataset_builder import CalvinDatasetBuilder

# add calvin_env to path
sys.path.append(str(Path(__file__).absolute().parents[2] / "models" / "flower_vla_calvin" / "calvin_env"))


class CalvinPolicyDatasetBuilder(CalvinDatasetBuilder):
    def __init__(self, timestamp, dataset_path, output_dir,
                 traj_simplification_rdp_epsilon=0.01, traj_drawing_thickness=2,
                 gif_frame_quantization_method=2, gif_frame_quantization_kmeans=1, gif_duration=67, gif_num_loops=0):
        
        self.timestamp = timestamp
        super().__init__(dataset_path, traj_simplification_rdp_epsilon)
        self.output_dir = output_dir
        self.traj_drawing_thickness = traj_drawing_thickness # pixels
        self.traj_drawing_circle_radius = 2 * traj_drawing_thickness # pixels
        self.gif_frame_quantization_method = gif_frame_quantization_method # enum of size 4, 2 = Image.FASTOCTREE (fast but less accurate)
        self.gif_frame_quantization_kmeans = gif_frame_quantization_kmeans # cluster changes of pixels allowed per kmeans iteration => lower = try harder to find best color palette, 0 = no clustering
        self.gif_duration = gif_duration # ms per frame => 67 ≈ 15 fps
        self.gif_num_loops = gif_num_loops # 0 = infinite

        os.makedirs(f"{self.output_dir}/{self.timestamp}", exist_ok=True)

        _file_handler = logging.FileHandler(f"{self.output_dir}/{self.timestamp}/build_dataset.log", mode='w')
        _file_handler.setLevel(logging.INFO)
        _file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s"))
        self._logger.addHandler(_file_handler)

        self.save_first_img_per_seq = False
        self.save_gif_per_seq = False

        self._logger.info(f"Initialized CalvinPolicyDatasetBuilder")


    def _draw_trajectory_onto_img(self, img, gripper_centers, gripper_widths):
        # gripper centers in image space => 2D

        img_copy = img.copy()

        is_gripper_open = gripper_widths[0] == self.CALVIN_GRIPPER_WIDTH_OPEN

        for i in range(len(gripper_centers) - 1):
            color = (round((i+1) / len(gripper_centers) * 255), 0, 0) # black to red over time
            cv2.line(img_copy, tuple(gripper_centers[i]), tuple(gripper_centers[i+1]), color, thickness=self.traj_drawing_thickness)

            if is_gripper_open and gripper_widths[i] == self.CALVIN_GRIPPER_WIDTH_CLOSED:
                # close gripper => green circle
                cv2.circle(img_copy, tuple(gripper_centers[i]), radius=self.traj_drawing_circle_radius, color=(0, 255, 0), thickness=self.traj_drawing_thickness)
                is_gripper_open = False
            elif not is_gripper_open and gripper_widths[i] == self.CALVIN_GRIPPER_WIDTH_OPEN:
                # open gripper => blue circle
                cv2.circle(img_copy, tuple(gripper_centers[i]), radius=self.traj_drawing_circle_radius, color=(0, 0, 255), thickness=self.traj_drawing_thickness)
                is_gripper_open = True

        return img_copy


    def build_trajectory_representation(self, gripper_centers_world, gripper_widths):
        # project simplified trajectory to image spaces & draw on images

        traj_imgs_seq = {"rgb_static": [], "rgb_gripper": []}
        for cam_id, cam_name in enumerate(["rgb_static", "rgb_gripper"]):
            for timestep in range(len(self.curr_seq["obs"]["robot_obs"])):
                # project gripper centers to both cams for first timestep
                # only update gripper cam for each timestep (only gripper cam moves)
                if timestep == 0 or cam_name == "rgb_gripper":
                    self.env.step(self.curr_seq["obs"]["rel_actions"][timestep]) # move gripper to position at timestep to update camera view matrix
                    simplified_gripper_centers_projected = self._project_gripper_centers_to_cam(gripper_centers_world, cam_id)

                img = self.curr_seq["obs"][cam_name][timestep]
                img_with_traj = self._draw_trajectory_onto_img(img, simplified_gripper_centers_projected, gripper_widths)

                traj_imgs_seq[cam_name].append(img_with_traj)

                if not self.save_gif_per_seq:
                    # only first frame of each sequence needed
                    break

        return {"traj_imgs_seq": traj_imgs_seq}


    def _save_imgs_gifs_to_disk(self, task_all_seqs, imgs_all_seqs, dataset_split):
        assert len(task_all_seqs) == len(imgs_all_seqs)

        content_to_be_saved = []
        if self.save_first_img_per_seq:
            imgs_dir = f"{self.output_dir}/imgs/{dataset_split}"
            os.makedirs(imgs_dir, exist_ok=True)
            content_to_be_saved.append("images")
        if self.save_gif_per_seq:
            gifs_dir = f"{self.output_dir}/gifs/{dataset_split}"
            os.makedirs(gifs_dir, exist_ok=True)
            content_to_be_saved.append("gifs")
        seq_loop_tqdm_desc = "Saving " + " and ".join(content_to_be_saved) + " to disk"
        
        num_digits = len(str(len(task_all_seqs)))

        for i, (task_seq, imgs_seq) in tqdm(enumerate(zip(task_all_seqs, imgs_all_seqs)), total=len(task_all_seqs), desc=seq_loop_tqdm_desc):
            for cam_name in ["rgb_static", "rgb_gripper"]:
                file_name = f"{i:0{num_digits}d}_{task_seq}_{cam_name.split('_')[1]}"

                if self.save_first_img_per_seq:
                    first_img = Image.fromarray(imgs_seq[cam_name][0])

                    first_img.save(f"{imgs_dir}/{file_name}.png")

                if self.save_gif_per_seq:
                    gif_frames = []
                    for img in imgs_seq[cam_name]:
                        # to reduce file size
                        quantized_img = Image.fromarray(img).quantize(method=self.gif_frame_quantization_method, kmeans=self.gif_frame_quantization_kmeans)

                        gif_frames.append(quantized_img)

                    gif_frames[0].save(f"{gifs_dir}/{file_name}.gif", save_all=True, append_images=gif_frames[1:], duration=self.gif_duration, loop=self.gif_num_loops)
        
        self._logger.info(f"Saved {len(imgs_all_seqs)} {' and '.join(content_to_be_saved)} to disk")


    def _embed_and_save_trajectory_imgs(self, task_all_seqs, traj_imgs_all_seqs, dataset_split):
        assert len(task_all_seqs) == len(traj_imgs_all_seqs), f"len(task_all_seqs) ({len(task_all_seqs)}) != len(traj_imgs_all_seqs) ({len(traj_imgs_all_seqs)})"

        # build auto_vis_lang_ann.npy (load auto_lang_ann and add vision annotations)
        auto_lang_ann = np.load(f"{self.dataset_path}/{dataset_split}/{self.AUTO_LANG_ANN_FOLDER}/auto_lang_ann.npy", allow_pickle=True).item()
        auto_vis_lang_ann = {"vision": {"ann": [], "emb": []}, "language": auto_lang_ann["language"], "info": auto_lang_ann["info"]}
        for traj_imgs_seq in tqdm(traj_imgs_all_seqs, total=len(traj_imgs_all_seqs), desc=f"Embedding trajectory images for {dataset_split} split"):
            first_static_traj_img = traj_imgs_seq["rgb_static"][0] # don't use rgb_gripper imgs as they don't show the traj well
            
            # don't embed trajectory image as it is not used for training
            
            auto_vis_lang_ann["vision"]["ann"].append(first_static_traj_img)
        
        auto_vis_lang_ann["vision"]["ann"] = np.stack(auto_vis_lang_ann["vision"]["ann"])
        
        vis_lang_ann_output_dir = f"{self.output_dir}/{self.timestamp}/{self.AUTO_VIS_LANG_ANN_FOLDER}/{dataset_split}"
        os.makedirs(vis_lang_ann_output_dir, exist_ok=True)
        np.save(f"{vis_lang_ann_output_dir}/auto_vis_lang_ann.npy", auto_vis_lang_ann)

        if dataset_split == "validation":
            # copy embeddings.npy from original dataset (saving vision embeddings doesn't make sense as scene varies & only one embedding per task)
            shutil.copy(f"{self.dataset_path}/{dataset_split}/{self.AUTO_LANG_ANN_FOLDER}/embeddings.npy", f"{vis_lang_ann_output_dir}/embeddings.npy")

            self._logger.info(f"Built vision-language annotation & embedding files for {dataset_split} split")
        else:
            self._logger.info(f"Built vision-language annotation file for {dataset_split} split")


    def build_dataset(self):
        for dataset_split in ["training", "validation"]:
            task_all_seqs, _, _, _, traj_imgs_all_seqs = self._build_trajectories(dataset_split)

            if self.save_first_img_per_seq or self.save_gif_per_seq:
                self._save_imgs_gifs_to_disk(task_all_seqs, traj_imgs_all_seqs, dataset_split)
            else:
                self._logger.info("Saving images or gifs to disk not requested by user")

            self._embed_and_save_trajectory_imgs(task_all_seqs, traj_imgs_all_seqs, dataset_split)
        
        self._logger.info("Finished building Policy dataset")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str, default="/DATA/calvin/task_ABC_D")
    parser.add_argument("--output-dir", type=str, default="/home/troth/data/iTRAP-flower/calvin_policy_dataset")
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    calvin_policy_dataset_builder = CalvinPolicyDatasetBuilder(timestamp=timestamp, dataset_path=args.dataset_path, output_dir=args.output_dir,
                                                               traj_drawing_thickness=5)
    calvin_policy_dataset_builder.build_dataset()
