import os
import sys
import logging
import hydra
import numpy as np
from pathlib import Path
from rdp import rdp
from omegaconf import OmegaConf
from tqdm import tqdm
from abc import ABC, abstractmethod

from calvin_dataloader import CalvinDataLoader


class CalvinDatasetBuilder(ABC):
    # don't touch
    CALVIN_GRIPPER_WIDTH_OPEN = 1.0
    CALVIN_GRIPPER_WIDTH_CLOSED = -1.0
    CLIP_VIS_CFG_PATH = "models/MoDE_Diffusion_Policy/conf/model/mode_agent.yaml"
    AUTO_LANG_ANN_FOLDER = "lang_clip_resnet50"
    AUTO_VIS_LANG_ANN_FOLDER = "vis_lang_clip_vit-b16_resnet50"


    def __init__(self, dataset_path, traj_simplification_rdp_epsilon):
        self.dataset_path = dataset_path
        os.makedirs(dataset_path, exist_ok=True)

        self.traj_simplification_rdp_epsilon = traj_simplification_rdp_epsilon # unit of world coordinates (meters?) => 0.01 = 1 cm?, 0.01 ≈ 6.8 points per trajectory
                                                                               # TODO: figure out world coords unit

        self.env = None
        self.dataloader = None
        self.curr_seq = None

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        _console_handler = logging.StreamHandler(stream=sys.stdout)
        _console_handler.setLevel(logging.INFO)
        _console_handler.setFormatter(logging.Formatter("\033[92m[%(asctime)s][%(levelname)s]\033[0m %(message)s"))
        self._logger.addHandler(_console_handler)


    @abstractmethod
    def build_trajectory_representation(self, gripper_centers_world, gripper_widths):
        pass


    def _simplify_trajectory(self, gripper_centers, gripper_widths):
        # Ramer-Douglas-Peucker algorithm

        assert len(gripper_centers) == len(gripper_widths)

        simplification_mask = rdp(gripper_centers, epsilon=self.traj_simplification_rdp_epsilon, return_mask=True)

        gripper_centers_simplified = gripper_centers[simplification_mask]
        gripper_widths_simplified = gripper_widths[simplification_mask]

        return gripper_centers_simplified, gripper_widths_simplified


    def _project_gripper_centers_to_cam(self, gripper_centers_world, cam_id):
        # TODO: handle rare massive outliers in gripper cam

        if cam_id == 1:
            # fix different names of projection & view matrices between static & gripper cam
            self.env.cameras[cam_id].projectionMatrix = self.env.cameras[cam_id].projection_matrix
            del self.env.cameras[cam_id].projection_matrix
            self.env.cameras[cam_id].viewMatrix = self.env.cameras[cam_id].view_matrix
            del self.env.cameras[cam_id].view_matrix

        gripper_centers_world_ones = np.c_[np.array(gripper_centers_world), np.ones(len(gripper_centers_world))]
        gripper_centers_projected = self.env.cameras[cam_id].project(gripper_centers_world_ones.T)

        return np.transpose(gripper_centers_projected)


    def _build_trajectories(self, dataset_split):
        env_conf = OmegaConf.load(f"{self.dataset_path}/{dataset_split}/.hydra/merged_config.yaml")
        del env_conf.cameras["tactile"] # not relevant for the Policy & VLM datasets and breaks hydra instantiation
        self.env = hydra.utils.instantiate(env_conf.env, use_vr=False, use_scene_info=True)

        calvin_root = Path(__file__).absolute().parents[2] / "models" / "MoDE_Diffusion_Policy" / "calvin_env"
        self.dataloader = CalvinDataLoader(calvin_root, dataset_path=f"{self.dataset_path}/{dataset_split}", annotations_folder=self.AUTO_LANG_ANN_FOLDER)
        
        lengths_simplified_trajs = []

        task_all_seqs = []
        traj_strings_all_seqs = []
        start_imgs_all_seqs = []
        traj_imgs_all_seqs = []
        for i, seq in tqdm(enumerate(self.dataloader), total=len(self.dataloader), desc=f"Building trajectories for {dataset_split} split"):
            assert len(seq["obs"]["robot_obs"]) == len(seq["obs"]["rel_actions"]) == len(seq["obs"]["rgb_static"]) == len(seq["obs"]["rgb_gripper"])

            self.curr_seq = seq # not pretty but required for build_trajectory_representation()

            task_all_seqs.append(self.curr_seq["anno"])

            # reset env to start of sequence
            self.env.reset(robot_obs=self.curr_seq["obs"]["robot_obs"][0], scene_obs=self.curr_seq["obs"]["scene_obs"][0])

            # get gripper centers & widths from sequence
            gripper_centers_world = np.array(self.curr_seq["obs"]["robot_obs"])[:, :3]
            gripper_widths = np.array(self.curr_seq["obs"]["robot_obs"])[:, -1]

            # simplify trajectory of center points in world space
            simplified_gripper_centers_world, simplified_gripper_widths = self._simplify_trajectory(gripper_centers_world, gripper_widths)
            lengths_simplified_trajs.append(len(simplified_gripper_centers_world))

            # build trajectory representation (trajectory images or trajectory string & start images)
            traj_representations = self.build_trajectory_representation(simplified_gripper_centers_world, simplified_gripper_widths)
            if traj_representations.keys() == {"traj_imgs_seq"}:
                traj_imgs_all_seqs.append(traj_representations["traj_imgs_seq"])
            elif traj_representations.keys() == {"traj_string_seq", "start_imgs_seq"}:
                traj_strings_all_seqs.append(traj_representations["traj_string_seq"])
                start_imgs_all_seqs.append(traj_representations["start_imgs_seq"])

        self._logger.info(f"Built {i+1} trajectories for {dataset_split} split, average trajectory length: {np.mean(lengths_simplified_trajs)}")

        return task_all_seqs, traj_strings_all_seqs, start_imgs_all_seqs, traj_imgs_all_seqs
