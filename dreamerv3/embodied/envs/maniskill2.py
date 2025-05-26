import functools

import elements
import embodied
import gym
import numpy as np

import numpy as np
import numpy as np
from typing import Dict, List

import numpy as np

from mani_skill2.sensors.camera import Camera
from mani_skill2.utils import common
import mani_skill2.envs
from mani_skill2.utils.sapien_utils import look_at

class FromManiskill2(embodied.Env):
  def __init__(self, 
               env, 
               obs_mode='pointcloud+rgb', 
               control_mode=None, num_envs=1, size=(128,128), 
               cam_name='base_camera',
               depth_max_mm=1000,
               obs_frame='base_pose',
               n_downsample_pts=512,
               use_segmented_pts=False,
               repeat=1,
               pose=None,
               **env_args
               ):
    # TODO seeding for reproducibility
    '''
    obs_mode : See https://maniskill.readthedocs.io/en/latest/user_guide/concepts/observation.html
      'pointcloud', 'rgb+depth', 'state_dict', 'state' (flattened), 'sensor_data' 
    control_mode : None will indicate pd_joint_delta_pos in PegInsertionSide-v1, 
      ex) 'pd_joint_delta_pos', 'pd_joint_pos', 'pd_ee_delta_pos', 'pd_ee_delta_pose', 
      'pd_ee_pose', 'pd_joint_target_delta_pos', 'pd_ee_target_delta_pos', 'pd_ee_target_delta_pose', 
      'pd_joint_vel', 'pd_joint_pos_vel', 'pd_joint_delta_pos_vel'
    num_envs : TODO / will parallelization benefit?
    '''
    self.use_pcd = False
    self.use_colored_pcd = False
    self.obs_frame = obs_frame
    self.n_downsample_pts=n_downsample_pts
    self.use_segmented_pts = use_segmented_pts

    if pose is None:
        pose = look_at([-0.5, -0.5, -0.5], [-0.1, 0, 0.1])

    if 'pointcloud' in obs_mode:
      self.use_pcd = True
      if obs_mode == 'pointcloud+rgb':
        self.use_colored_pcd = True
      obs_mode = 'pointcloud' # this detours using pointcloud wrapper of maniskill

    name = env
    self._name = name
    self.model_ids = None
    if "TurnFaucet" in name:
        self.model_ids = ["5002", "5004", "5005", "5007"]
    elif "PushChair" in name:
        self.model_ids = ["3001", "3003", "3005", "3008", 
                            "3010", "3013", "3016", "3020",
                            "3021", "3022", "3024", "3025", 
                            "3027"]
    elif "OpenCabinetDrawer" in name:
        self.model_ids = ["1067"]

    device = "cuda"
    if "v0" in name:
        if self.model_ids is not None:
            env = gym.make(
                name,
                obs_mode=obs_mode,
                control_mode=control_mode,
                model_ids=self.model_ids,
                pose=pose
            )
            print(
                f"Env. Name: {name}, Observation Mode: {obs_mode}, Control Mode: {control_mode}, Model Ids: {self.model_ids}"
            )
        else:
            print(env_args)
            print(f"name: {name}")
            if name in ["StackCubeEval-v0", "LiftRedCube-v0", "Habitat-v0"]:
                env = gym.make(name, obs_mode=obs_mode, control_mode=control_mode, pose=pose, renderer_kwargs={"device": device}, **env_args)
            else:
                env = gym.make(name, obs_mode=obs_mode, control_mode=control_mode, renderer_kwargs={"device": device}, **env_args)
            print(
                f"Env. Name: {name}, Observation Mode: {obs_mode}, Control Mode: {control_mode}"
            )
    elif "v1" in name:
        if self.model_ids is not None:
            env = gym.make(name, obs_mode=obs_mode, model_ids=self.model_ids, renderer_kwargs={"device": device},)
            print(
                f"Env. Name: {name}, Observation Mode: {obs_mode}, Control Mode: {env.control_mode}, Model Ids: {self.model_ids}"
            )
        else:
            env = gym.make(name, obs_mode=obs_mode, renderer_kwargs={"device": device},)
            print(
                f"Env. Name: {name}, Observation Mode: {obs_mode}, Control Mode: {env.control_mode}"
            )
    else:
        raise NotImplemented("Version should be either v0 or v1.")
    self._env = env

    self.size = size
    # self._env = CPUGymWrapper(self._env)
    if obs_mode == 'rgbd':
      obs_mode = 'rgb+depth'
    self._obs_mode = obs_mode
    # self._env = CPUGymWrapper(self._env) # Make this parallel?

    self.cam_name = cam_name
    self.depth_max_mm = depth_max_mm

    if type(self.cam_name) != str and "rgb" in obs_mode:
      raise ValueError("Only single camera is supported for RGB observation.")
      # for pointclouds, data is automatically gained data using all cameras
      # , so need to modify default setup
    self._obs_key = obs_mode

    self._act_key = 'action'

    self._obs_dict = hasattr(self._env.observation_space, 'spaces')
    self._act_dict = hasattr(self._env.action_space, 'spaces')
    self._action_repeat = repeat
    self._done = True
    self._info = None

  @property
  def env(self):
    return self._env

  @property
  def info(self):
    return self._info

  @functools.cached_property
  def obs_space(self):
    spaces = {}
    if 'state' in self._obs_key:
      # for key, value in self._env.observation_space['agent'].spaces.items():
      # breakpoint()
      # tot_dim = sum([self._env.observation_space['agent'][k].shape[0] for k in self._env.observation_space['agent'].keys()])
      spaces['state'] = embodied.Space(np.float32, self._env.observation_space['state'].shape)
  
    if 'pointcloud' in self._obs_key:
      pcd_dim = 7 if self.use_colored_pcd else 4
      # spaces["pointcloud"] = embodied.Space(np.float32, (self.num_points, pcd_dim))
      # spaces['pointcloud'] = embodied.Space(np.float32, (self._env.observation_space['pointcloud']['xyzw'].shape[0], pcd_dim))
      spaces['pointcloud'] = embodied.Space(np.float32, (self.size[0]*self.size[1]*2, pcd_dim))
      # spaces['render_image'] = embodied.Space(np.uint8, (self.size[0], self.size[1], 3))
      # spaces['world_pointcloud'] = embodied.Space(np.float32, (self.size[0]*self.size[1], 4))
      if self.obs_frame == 'tcp_pose':
        spaces['obs_frame_pose'] = embodied.Space(np.float32, (7,))
      # spaces['raw_pointcloud'] = embodied.Space(np.float32, (self.size[0]*self.size[1], pcd_dim))

    # if 'rgb' in self._obs_key and 'pointcloud' not in self._obs_key:
    #   if 'depth' not in self._obs_key:
    #     spaces['image'] = embodied.Space(np.uint8, self._env.observation_space['sensor_data'][self.cam_name]['rgb'].shape)
    #   else:
    #     spaces['image'] = embodied.Space(np.uint16, tuple(list(self._env.observation_space['sensor_data'][self.cam_name]['rgb'].shape[:-1]) + [4]))
    if 'rgb' in self._obs_key and 'pointcloud' not in self._obs_key:
      spaces['image'] = embodied.Space(np.uint8, self._env.observation_space['image'][self.cam_name]['rgb'].shape)
    
    if 'depth' in self._obs_key and 'pointcloud' not in self._obs_key:
        spaces['depth'] = embodied.Space(np.uint16, self._env.observation_space['image'][self.cam_name]['depth'].shape)

    return {
        **spaces,
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @functools.cached_property
  def act_space(self):
    if self._act_dict:
      spaces = self._flatten(self._env.action_space.spaces)
    else:
      spaces = {self._act_key: self._env.action_space}
    spaces = {k: self._convert(v) for k, v in spaces.items()}
    spaces['reset'] = embodied.Space(bool)
    return spaces

  def step(self, action):      
    if action['reset'] or self._done:
      self._done = False
      obs = self._env.reset() 
      # info : 'elapsed_steps', 'success', 'peg_head_pos_at_hole', 'reconfigure'
      return self._obs(obs, 0.0, is_first=True)
    if self._act_dict:
      action = self._unflatten(action)
    else:
      action = action[self._act_key]

    reward = 0.0
    for _ in range(self._action_repeat):
        obs, r, done, self._info = self._env.step(action)
        reward += r or 0.0
        if done:
          break

    terminated = False
    truncated = done

    self._done = done

    return self._obs(
        obs, reward,
        is_last=bool(truncated),
        is_terminal=bool(terminated))

  def _obs(
      self, obs, reward, is_first=False, is_last=False, is_terminal=False):

    obs_dict = {}

    if 'state' in self._obs_key:
      obs_dict['state'] = obs['state']

    if 'pointcloud' in self._obs_key:
        xyzw = obs['pointcloud']['xyzw']
        
        # 
        mask = np.linalg.norm(xyzw[..., :3], axis=-1) > self.depth_max_mm
        if self.use_segmented_pts:
            if "LiftCubeHabitat" in self._name:
               mask = mask | np.logical_and(xyzw[..., 0] < -0.5, xyzw[..., 1] < -0.5)
            else:
                mask = mask | (xyzw[..., 2] <= 1e-3)

        xyzw[mask, 3] = 0

        # pad when having insufficienet valid points
        valid_mask = xyzw[:, 3] > 0
        n_valid = int(valid_mask.sum())
        if n_valid < self.n_downsample_pts:
            n_needed = self.n_downsample_pts - n_valid
            valid_idx = np.flatnonzero(valid_mask)
            dup_idx   = np.random.choice(valid_idx, size=n_needed, replace=True)
            fill_pos  = np.flatnonzero(~valid_mask)[:n_needed]
            xyzw[fill_pos] = xyzw[dup_idx]

        if self.use_colored_pcd:
            obs_dict['pointcloud'] = np.concatenate([xyzw, obs['pointcloud']['rgb']], axis=-1)
        else:
            obs_dict['pointcloud'] = xyzw

    if 'pointcloud' not in self._obs_key and 'rgb' in self._obs_key:
      obs_dict['image'] = obs['image'][self.cam_name]['rgb']

    # if 'render_image' in obs:
    #   obs_dict['render_image'] = obs['render_image']

    if 'depth' in self._obs_key:
      depth = (obs['image'][self.cam_name]['depth']*1000).astype(np.uint16)
      depth = np.where(depth == 0, self.depth_max_mm, depth) 
      depth = depth.clip(0, self.depth_max_mm)
      obs_dict['depth'] = depth
      # obs_dict['image'] = np.concatenate([obs_dict['image'], depth], axis=-1)

    obs_dict = {k: np.asarray(v) for k, v in obs_dict.items()}
    obs_dict.update(
        reward=np.float32(reward),
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal)
    return obs_dict

  def render(self):
    image = self._env.render()
    assert image is not None
    return image

  def close(self):
    try:
      self._env.close()
    except Exception:
      pass

  def _flatten(self, nest, prefix=None):
    result = {}
    for key, value in nest.items():
      key = prefix + '/' + key if prefix else key
      if isinstance(value, gym.spaces.Dict):
        value = value.spaces
      if isinstance(value, dict):
        result.update(self._flatten(value, key))
      else:
        result[key] = value
    return result

  def _unflatten(self, flat):
    result = {}
    for key, value in flat.items():
      parts = key.split('/')
      node = result
      for part in parts[:-1]:
        if part not in node:
          node[part] = {}
        node = node[part]
      node[parts[-1]] = value
    return result

  def _convert(self, space):
    if hasattr(space, 'n'):
      return embodied.Space(np.int32, (), 0, space.n)
    return embodied.Space(space.dtype, space.shape, space.low, space.high)