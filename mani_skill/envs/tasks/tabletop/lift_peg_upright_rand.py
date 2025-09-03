from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import torch.random
from transforms3d.euler import euler2quat

from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.building import actors
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils.registration import register_env
from mani_skill.utils.sapien_utils import look_at
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array


@register_env("LiftPegUprightRand-v1", max_episode_steps=50)
class LiftPegUprightRandEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to move a peg laying on the table to any upright position on the table

    **Randomizations:**
    - the peg's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat along it's length on the table

    **Success Conditions:**
    - the absolute value of the peg's y euler angle is within 0.08 of $\pi$/2 and the z position of the peg is within 0.005 of its half-length (0.12).
    """

    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/LiftPegUpright-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda", "fetch"]
    agent: Union[Panda, Fetch]

    peg_half_width = 0.025
    peg_half_length = 0.12

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]


    @property
    def _default_human_render_camera_configs(self):

        pose = look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        cam_configs = [CameraConfig("render_camera", pose, 256, 256, 1, 0.01, 100)]

        target_bounds = [[-0.2, 0], [-0.1, 0.1], [-0.1, 0.1]]
        # randomly choose centers
        for i in range(600):
            azumith = np.random.uniform(-90, 90) / 180 * np.pi
            elevation = np.random.uniform(30, 60) / 180 * np.pi
            distance = np.random.uniform(0.45, .7)
            xyz = np.array([np.cos(azumith) * np.cos(elevation), np.sin(azumith) * np.cos(elevation), np.sin(elevation)]) * distance
            center = [np.random.uniform(target_bounds[0][0], target_bounds[0][1]),
                    np.random.uniform(target_bounds[1][0], target_bounds[1][1]),
                    np.random.uniform(target_bounds[2][0], target_bounds[2][1])]
            pose = look_at(xyz, center)
            cam_configs.append(CameraConfig(f"cam_{i}", pose, 256, 256, 1, 0.01, 100))

        return cam_configs


    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Always de-texture the ground grid (patternless white floor)
        for part in self.table_scene.ground._objs:
            comp = part.find_component_by_type(sapien.render.RenderBodyComponent)
            if comp is None:
                continue
            for shape in comp.render_shapes:
                for triangle in shape.parts:
                    triangle.material.set_base_color(np.array([1.0, 1.0, 1.0, 1.0]))
                    triangle.material.set_base_color_texture(None)
                    triangle.material.set_normal_texture(None)
                    triangle.material.set_emission_texture(None)
                    triangle.material.set_transmission_texture(None)
                    triangle.material.set_metallic_texture(None)
                    triangle.material.set_roughness_texture(None)

        # the peg that we want to manipulate
        self.peg = actors.build_twocolor_peg(
            self.scene,
            length=self.peg_half_length,
            width=self.peg_half_width,
            color_1=np.array([176, 14, 14, 255]) / 255,
            color_2=np.array([12, 42, 160, 255]) / 255,
            name="peg",
            body_type="dynamic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            # Randomize table yaw uniformly in [0, 2pi) each initialization (about +Z)
            angles = torch.rand((b,), device=self.device) * (2 * torch.pi) + np.pi / 2
            q = torch.zeros((b, 4), device=self.device)
            q[:, 0] = (angles / 2).cos()
            q[:, 3] = (angles / 2).sin()
            # Randomize table XY translation: x in [-0.2, 0.2] around base -0.12, y in [-0.2, 0.2]
            table_xy_shift = torch.rand((b, 2), device=self.device) * 0.4 - 0.2
            p = torch.zeros((b, 3), device=self.device)
            p[:, 0] = -0.12 + table_xy_shift[:, 0]
            p[:, 1] = table_xy_shift[:, 1]
            p[:, 2] = -0.9196429
            self.table_scene.table.set_pose(Pose.create_from_pq(p=p, q=q))

            xyz = torch.zeros((b, 3))
            # Center randomization at (-0.1, 0.0) with the same span (0.2)
            xyz[..., :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[..., 0] -= 0.1
            xyz[..., 2] = self.peg_half_width

            # Randomize peg yaw uniformly in [0, 2pi) while keeping it flat on the table
            peg_yaw = torch.rand((b,), device=self.device) * (2 * torch.pi)
            half_yaw_cos = (peg_yaw / 2).cos()
            half_yaw_sin = (peg_yaw / 2).sin()
            # Base rotation to lay the peg flat (Rx(pi/2))
            base_half = np.pi / 4
            base_c = torch.tensor(np.cos(base_half), device=self.device).repeat(b)
            base_s = torch.tensor(np.sin(base_half), device=self.device).repeat(b)
            # Quaternion multiplication: q_total = q_yaw(z, peg_yaw) * q_base(x, pi/2)
            q_total = torch.zeros((b, 4), device=self.device)
            q_total[:, 0] = half_yaw_cos * base_c
            q_total[:, 1] = half_yaw_cos * base_s
            q_total[:, 2] = half_yaw_sin * base_s
            q_total[:, 3] = half_yaw_sin * base_c

            obj_pose = Pose.create_from_pq(p=xyz, q=q_total)
            self.peg.set_pose(obj_pose)

    def evaluate(self):
        q = self.peg.pose.q
        qmat = rotation_conversions.quaternion_to_matrix(q)
        # Peg's axis is local +X → first column of rotation matrix in world frame
        # Check angle to world Z within 20°: |dot(axis, ez)| > cos(20°)
        axis_world_z_abs = torch.abs(qmat[:, 2, 0])
        upright_cos_thresh = torch.cos(torch.tensor(np.deg2rad(20.0), device=self.device))
        is_peg_upright = axis_world_z_abs > upright_cos_thresh

        # close_to_table = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length) < 0.005
        # return {
        #     "success": is_peg_upright & close_to_table,
        # }
        return {
            "success": is_peg_upright,
        }

    def _get_obs_extra(self, info: Dict):
        obs = dict(
            tcp_pose=self.agent.tcp.pose.raw_pose,
        )
        if self.obs_mode_struct.use_state:
            obs.update(
                obj_pose=self.peg.pose.raw_pose,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        # rotation reward as cosine similarity between peg direction vectors
        # peg center of mass to end of peg, (1,0,0), rotated by peg pose rotation
        # dot product with its goal orientation: (0,0,1) or (0,0,-1)
        qmats = rotation_conversions.quaternion_to_matrix(self.peg.pose.q)
        vec = torch.tensor([1.0, 0, 0], device=self.device)
        goal_vec = torch.tensor([0, 0, 1.0], device=self.device)
        rot_vec = (qmats @ vec).view(-1, 3)
        # abs since (0,0,-1) is also valid, values in [0,1]
        rot_rew = (rot_vec @ goal_vec).view(-1).abs()
        reward = rot_rew

        # position reward using common maniskill distance reward pattern
        # giving reward in [0,1] for moving center of mass toward half length above table
        z_dist = torch.abs(self.peg.pose.p[:, 2] - self.peg_half_length)
        reward += 1 - torch.tanh(5 * z_dist)

        # small reward to motivate initial reaching
        # initially, we want to reach and grip peg
        to_grip_vec = self.peg.pose.p - self.agent.tcp.pose.p
        to_grip_dist = torch.linalg.norm(to_grip_vec, axis=1)
        reaching_rew = 1 - torch.tanh(5 * to_grip_dist)
        # reaching reward granted if gripping block
        reaching_rew[self.agent.is_grasping(self.peg)] = 1
        # weight reaching reward less
        reaching_rew = reaching_rew / 5
        reward += reaching_rew

        reward[info["success"]] = 3
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 3.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward
