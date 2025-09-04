from typing import Any, Dict
from pathlib import Path
import os.path as osp

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder.table import scene_builder as table_scene_builder_module
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array, GPUMemoryConfig, SimConfig


class LargeTableSceneBuilder(TableSceneBuilder):
    """Env-local table builder with square tabletop 2.418 m x 2.418 m."""

    def __init__(self, env, robot_init_qpos_noise=0.02):
        super().__init__(env, robot_init_qpos_noise)

    def build(self):
        model_dir = Path(osp.dirname(table_scene_builder_module.__file__)) / "assets"
        table_model_file = str(model_dir / "table.glb")

        # Visual scaling: make world XY equal to 2.418 m each (accounting for 90° yaw)
        # local X -> world Y, local Y -> world X
        visual_scale_x = 2 * 1.75
        visual_scale_y = 1.75
        visual_scale_z = 1.75
        base_pose = sapien.Pose(p=[-0.12, 0, -0.9196429], q=euler2quat(0, 0, np.pi / 2))

        # 1) Collision-only actor (fixed)
        col_builder = self.scene.create_actor_builder()
        col_builder.add_box_collision(
            pose=sapien.Pose(p=[0, 0, 0.9196429 / 2]),
            half_size=(2.418 / 2, 2.418 / 2, 0.9196429 / 2),
        )
        col_builder.initial_pose = base_pose
        table_collision = col_builder.build_kinematic(name="table-collision")

        # 2) Visual-only actor (fixed)
        vis_fixed_builder = self.scene.create_actor_builder()
        vis_fixed_builder.add_visual_from_file(
            filename=table_model_file,
            scale=[visual_scale_x, visual_scale_y, visual_scale_z],
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        vis_fixed_builder.initial_pose = base_pose
        table_vis_fixed = vis_fixed_builder.build_kinematic(name="table-visual-fixed")

        # 3) Visual-only actor (randomized per-episode)
        vis_rand_builder = self.scene.create_actor_builder()
        vis_rand_builder.add_visual_from_file(
            filename=table_model_file,
            scale=[visual_scale_x, visual_scale_y, visual_scale_z],
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        )
        vis_rand_builder.initial_pose = base_pose
        table_vis_rand = vis_rand_builder.build_kinematic(name="table-visual-rand")

        # Reported dimensions
        self.table_length = 2.418
        self.table_width = 2.418
        self.table_height = 0.9196429

        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=-self.table_height
        )
        # Store references
        self.table = table_collision
        self.table_collision = table_collision
        self.table_vis_fixed = table_vis_fixed
        self.table_vis_rand = table_vis_rand
        self.scene_objects = [self.table_collision, self.table_vis_fixed, self.table_vis_rand, self.ground]

@register_env("RollBallRand-v1", max_episode_steps=80)
class RollBallRandEnv(BaseEnv):
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/RollBall-v1_rt.mp4"
    SUPPORTED_ROBOTS = ["panda"]

    agent: Panda

    goal_radius: float = 0.1
    ball_radius: float = 0.035
    reached_status: torch.Tensor

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, fixed: bool = False, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.fixed = fixed
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25, max_rigid_patch_count=2**18
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[-0.1, 0.9, 0.3], target=[0.0, 0.0, 0.0])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        # Follow Rand formats: many human cameras around a sphere, pointing to a fixed center
        center = [-0.1, 0.5, 0.0]
        cam_configs = [
            CameraConfig("render_camera", sapien_utils.look_at([-0.6, 1.3, 0.8], center), 256, 256, np.pi / 2, 0.01, 100)
        ]
        # Half arc behind the robot (excluding 80-100 deg): azimuth in [0, 180] deg, elevation in [30, 60] deg, distance in [0.45, 0.7]
        for i in range(600):
            az_deg = np.random.uniform(0, 160)
            if az_deg >= 80:
                az_deg += 20
            azumith = az_deg / 180 * np.pi
            elevation = np.random.uniform(30, 60) / 180 * np.pi
            distance = np.random.uniform(0.6, 0.7)
            direction = np.array(
                [
                    np.cos(azumith) * np.cos(elevation),
                    np.sin(azumith) * np.cos(elevation),
                    np.sin(elevation),
                ]
            )
            xyz = np.array(center) + direction * distance
            pose = sapien_utils.look_at(xyz, center)
            cam_configs.append(CameraConfig(f"cam_{i}", pose, 256, 256, np.pi / 2, 0.01, 100))
        return cam_configs

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = LargeTableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # Toggle visibility: when fixed=True show fixed visual, else show randomized visual
        # Hide by setting alpha to 0 (fully transparent)
        def _set_alpha(actor, alpha: float):
            for part in actor._objs:
                comp = part.find_component_by_type(sapien.render.RenderBodyComponent)
                if comp is None:
                    continue
                for shape in comp.render_shapes:
                    for tri in shape.parts:
                        c = np.array([1.0, 1.0, 1.0, alpha], dtype=np.float32)
                        tri.material.set_base_color(c)

        if self.fixed:
            _set_alpha(self.table_scene.table_vis_fixed, 1.0)
            _set_alpha(self.table_scene.table_vis_rand, 0.0)
        else:
            _set_alpha(self.table_scene.table_vis_fixed, 0.0)
            _set_alpha(self.table_scene.table_vis_rand, 1.0)

        # De-texture the ground grid (patternless white floor) unless fixed
        if not self.fixed:
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

        self.ball = actors.build_sphere(
            self.scene,
            radius=self.ball_radius,
            color=[0, 0.2, 0.8, 1],
            name="ball",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )

        self.goal_region = actors.build_red_white_target(
            self.scene,
            radius=self.goal_radius,
            thickness=1e-5,
            name="goal_region",
            add_collision=False,
            body_type="kinematic",
            initial_pose=sapien.Pose(p=[0, 0, 0.1]),
        )
        self.reached_status = torch.zeros(self.num_envs, dtype=torch.float32)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self.reached_status = self.reached_status.to(self.device)
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)

            table_xy_shift = (torch.rand((b, 2), device=self.device) * 2.0 - 1.0) * 0.2
            p = torch.zeros((b, 3), device=self.device)
            p[:, 0] = -0.12 + table_xy_shift[:, 0]
            p[:, 1] = 0.0 + table_xy_shift[:, 1] 
            p[:, 2] = -0.9196429

            yaw = torch.rand((b,), device=self.device) * (2 * torch.pi)
            q = torch.zeros((b, 4), device=self.device)
            q[:, 0] = (yaw / 2 + torch.pi / 4).cos()
            q[:, 3] = (yaw / 2 + torch.pi / 4).sin()
            self.table_scene.table_vis_rand.set_pose(Pose.create_from_pq(p=p, q=q))

            robot_pose = Pose.create_from_pq(
                p=[-0.1, 1.0, 0], q=[0.7071, 0, 0, -0.7072]
            )
            self.agent.robot.set_pose(robot_pose)

            xyz = torch.zeros((b, 3))
            xyz[..., 0] = (torch.rand((b)) * 2 - 1) * 0.3 - 0.1
            xyz[..., 1] = torch.rand((b)) * 0.2 + 0.5
            xyz[..., 2] = self.ball_radius
            q = [1, 0, 0, 0]

            obj_pose = Pose.create_from_pq(p=xyz, q=q)
            self.ball.set_pose(obj_pose)

            xyz_goal = torch.zeros((b, 3))
            xyz_goal[..., 0] = (torch.rand((b)) * 2 - 1) * 0.3 - 0.1
            # Target Y in [0.0, 0.2]
            xyz_goal[..., 1] = torch.rand((b)) * 0.2 - 0.3
            xyz_goal[..., 2] = 1e-3
            self.goal_region.set_pose(
                Pose.create_from_pq(
                    p=xyz_goal,
                    q=euler2quat(0, np.pi / 2, 0),
                )
            )
        self.reached_status[env_idx] = 0.0

    def evaluate(self):
        is_obj_placed = (
            torch.linalg.norm(
                self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
            )
            < self.goal_radius
        )
        return {"success": is_obj_placed}

    def _get_obs_extra(self, info: Dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                goal_pos=self.goal_region.pose.p,
                ball_pose=self.ball.pose.raw_pose,
                ball_vel=self.ball.linear_velocity,
                tcp_to_ball_pos=self.ball.pose.p - self.agent.tcp.pose.p,
                ball_to_goal_pos=self.goal_region.pose.p - self.ball.pose.p,
            )
        return obs

    def compute_dense_reward(self, obs: Any, action: Array, info: Dict):
        unit_vec = self.ball.pose.p - self.goal_region.pose.p
        unit_vec = unit_vec / torch.linalg.norm(unit_vec, axis=1, keepdim=True)
        tcp_hit_pose = Pose.create_from_pq(
            p=self.ball.pose.p + unit_vec * (self.ball_radius + 0.05),
        )
        tcp_to_hit_pose = tcp_hit_pose.p - self.agent.tcp.pose.p
        tcp_to_hit_pose_dist = torch.linalg.norm(tcp_to_hit_pose, axis=1)
        self.reached_status[tcp_to_hit_pose_dist < 0.04] = 1.0
        reaching_reward = 1 - torch.tanh(2 * tcp_to_hit_pose_dist)

        obj_to_goal_dist = torch.linalg.norm(
            self.ball.pose.p[..., :2] - self.goal_region.pose.p[..., :2], axis=1
        )

        reached_reward = 1 - torch.tanh(obj_to_goal_dist)

        reward = (
            20 * reached_reward * self.reached_status
            + reaching_reward * (1 - self.reached_status)
            + self.reached_status
        )

        reward[info["success"]] = 30.0
        return reward

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: Dict):
        max_reward = 30.0
        return self.compute_dense_reward(obs=obs, action=action, info=info) / max_reward




    


