# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 19:30:05 2025

@author: Jean-Baptiste Bouvier

Made-up car environment built from the rover.xml of 
https://github.com/griloHBG/Rover4We
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from scipy.spatial.transform import Rotation as R

from gymnasium import utils
from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv

DEFAULT_CAMERA_CONFIG = {"distance": 2.0}

def norm(x):
    return np.linalg.norm(x)


class CarEnv(MujocoEnv, utils.EzPickle):
    metadata = { "render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 20}

    """    
    | Num | Action                      |      Min      |     Max     |     Name    | Joint | Unit         |
    |-----|-----------------------------|---------------|-------------|-------------|-------|--------------|
    | 0   | Steering                    | -1 (right)    | 1 (left)    | ghost_steer | hinge | torque (N m) |
    | 1   | Wheel angular acceleration  | -1 (backward) | 1 (forward) | drive       | hinge | torque (N m) |
   

    | Num | qpos                                    | Min  | Max |        Name       | Joint | Unit         |
    | --- | ----------------------------------------| ---- | --- | ----------------- | ----- | ------------ |
    | 0   | x-coordinate of the car                 | -Inf | Inf | rootx             | slide | position (m) |
    | 1   | y-coordinate of the car                 | -Inf | Inf | rooty             | slide | position (m) |
    | 2   | z-coordinate of the car                 | -Inf | Inf | rootz             | slide | position (m) |
    | 3   | quaternion car  qw                      | -Inf | Inf | quat w            | hinge | quaternion   |
    | 4   | quaternion car  qx                      | -Inf | Inf | quat x            | hinge | quaternion   |
    | 5   | quaternion car  qy                      | -Inf | Inf | quat y            | hinge | quaternion   |
    | 6   | quaternion car  qz                      | -Inf | Inf | quat z            | hinge | quaternion   |
    | 7   | angle of the rear left wheel            | -Inf | Inf | r-l-drive-hinge   | hinge | angle (rad)  |
    | 8   | angle of the rear right wheel           | -Inf | Inf | r-r-drive-hinge   | hinge | angle (rad)  |
    | 9   | steering angle                          | -Inf | Inf | ghost-steer-hinge | hinge | angle (rad)  |
    | 10  | steering angle of the front left wheel  | -Inf | Inf | f-l-steer-hinge   | hinge | angle (rad)  |
    | 11  | angle of the front left wheel 1         | -Inf | Inf | f-l-drive-hinge-1 | hinge | angle (rad)  |
    | 12  | angle of the front left wheel 2         | -Inf | Inf | f-l-drive-hinge-2 | hinge | angle (rad)  |
    | 13  | steering angle of the front right wheel | -Inf | Inf | f-r-steer-hinge   | hinge | angle (rad)  |
    | 14  | angle of the front right wheel 1        | -Inf | Inf | f-r-drive-hinge-1 | hinge | angle (rad)  |
    | 15  | angle of the front right wheel 2        | -Inf | Inf | f-r-drive-hinge-2 | hinge | angle (rad)  |
     
    | Num | qvel                                               | Min  | Max |        Name       | Joint |           Unit           |
    | --- | ---------------------------------------------------| ---- | --- | ----------------- | ----- | ------------------------ |
    | 0   | x-velocity of the car                              | -Inf | Inf | rootx             | slide | linear velocity (m/s)    |
    | 1   | y-velocity of the car                              | -Inf | Inf | rooty             | slide | linear velocity (m/s)    |
    | 2   | z-velocity of the car                              | -Inf | Inf | rootz             | slide | linear velocity (m/s)    |
    | 3   | x-angular velocity of the car                      | -Inf | Inf | quat x            | hinge | angular velocity (rad/s) |
    | 4   | y-angular velocity of the car                      | -Inf | Inf | quat y            | hinge | angular velocity (rad/s) |
    | 5   | z-angular velocity of the car                      | -Inf | Inf | quat z            | hinge | angular velocity (rad/s) |
    | 6   | angular velocity of the rear left wheel            | -Inf | Inf | r-l-drive-hinge   | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of the rear right wheel           | -Inf | Inf | r-r-drive-hinge   | hinge | angular velocity (rad/s) |
    | 8   | steering angular velocity                          | -Inf | Inf | ghost-steer-hinge | hinge | angular velocity (rad/s) |
    | 9   | steering angular velocity of the front left wheel  | -Inf | Inf | f-l-steer-hinge   | hinge | angular velocity (rad/s) |
    | 10  | angular velocity of the front left wheel 1         | -Inf | Inf | f-l-drive-hinge-1 | hinge | angular velocity (rad/s) |
    | 11  | angular velocity of the front left wheel 2         | -Inf | Inf | f-l-drive-hinge-2 | hinge | angular velocity (rad/s) |
    | 12  | steering angular velocity of the front right wheel | -Inf | Inf | f-r-steer-hinge   | hinge | angular velocity (rad/s) |
    | 13  | angular velocity of the front right wheel 1        | -Inf | Inf | f-r-drive-hinge-1 | hinge | angular velocity (rad/s) |
    | 14  | angular velocity of the front right wheel 2        | -Inf | Inf | f-r-drive-hinge-2 | hinge | angular velocity (rad/s) |
    

    """
    def __init__(self, render_mode="rgb_array", verbose=False, 
                 reduced_model=False, **kwargs):
        
        self.name = "Car"
        self.reduced_model = reduced_model
        if reduced_model:
            self.state_size = 6 # only x, y, car- and their derivatives
        else:
            self.state_size = 31 
            
        self.action_size = 2
        self.action_max = np.array([ 1.,  1.])
        self.action_min = np.array([-1., -1.])
        self.verbose = verbose
        xml_file = "~/Documents/PythonScripts/WaymoCar/car.xml" # Update this path to your repo
        frame_skip = 2
        self.reset_qpos = np.array([0., 0., 0.1993, # x y z
                                    1., 0., 6e-4, 0, # qw qx qy qz
                                    0., 0., 0.,    # rear left right wheels drive, steering
                                    0., 0.1, 0.1,  # front left  wheel steering, drive-1, drive-2
                                    0., 0.1, 0.1]) # front right wheel steering, drive-1, drive-2
        
        self.max_episode_steps = 200
        
        observation_space = Box(low=-np.inf, high=np.inf, shape=(self.state_size,),
                                dtype=np.float64)
        self._ctrl_cost_weight = 0.02
        self.reward_threshold = 60
        self._reset_noise_scale = 0.1
        self._states_to_noise = [0, 1, 7, 8, 11, 12, 14, 15] # apply random noise at reset on these qpos, don't mess with steering
        
        MujocoEnv.__init__(self, xml_file,  frame_skip,
                           observation_space=observation_space,
                           default_camera_config=DEFAULT_CAMERA_CONFIG,
                           render_mode=render_mode, **kwargs)


        ### Iterate over the geoms named "obstacle_{i}" in xml, assuming all cylinders
        self.obstacle_xy = []
        self.obstacle_radius = []
        obstacle_id = 1
        done = False
        while not done:
            try:
               self.obstacle_xy.append( self.model.geom(f"obstacle_{obstacle_id}").pos[:2] )
               self.obstacle_radius.append( self.model.geom(f"obstacle_{obstacle_id}").size[0] )
               obstacle_id += 1
            except:
               done = True
        self.obstacle_xy = np.vstack(self.obstacle_xy)
        self.obstacle_radius = np.vstack(self.obstacle_radius)
        self.num_obstacles = self.obstacle_xy.shape[0]
        
        ### delimit the 4 corners of the car
        x, y = self.model.geom("chassis").size[:2]/2
        self.body_vertices = np.array([[x+y, y], [x+y, -y], [-x-y, y], [-x-y, -y]])

        ### State space bounds
        y_min = self.obstacle_xy[:,1].min() - self.obstacle_radius[0,0]
        y_max = self.obstacle_xy[:,1].max() + self.obstacle_radius[0,0]
        self.xy_bounds = np.array([[-np.inf, y_min],
                                   [ np.inf, y_max]])


    def _obstacle_collision(self):
        """Checks whether the car collides with any of the obstacles"""
        # Rotate the car vertices to match the orientation of the car
        vertices = self.body_vertices @ R.from_quat(self.data.qpos[3:7].copy(), scalar_first=True).as_matrix()[:2, :2].T # rotate around z 
        vertices += self.data.qpos[:2].copy()
        for obst_id in range(self.num_obstacles):
            for v_id in range(4):
                if norm(self.obstacle_xy[obst_id] - vertices[v_id]) < self.obstacle_radius[obst_id]:
                    if self.verbose:
                        print(f"Collision with obstacle_{obst_id+1}")
                    return True
        return False # no collision if we arrived at this point
    
    
    def _out_of_bounds(self):
        """Checks whether the car is out of bounds"""
        xy = self.data.qpos[:2].copy()
        out = (xy < self.xy_bounds[0]).any() or (xy > self.xy_bounds[1]).any()
        if out and self.verbose:
            print(f"State [{xy[0]:.1f}, {xy[1]:.1f}] is out of bounds")
        return out
        
    
    def step(self, action):
        
        self.do_simulation(action, self.frame_skip)
        obs = self.get_state()
        
        self.episode_step += 1
        terminated = self.episode_step > self.max_episode_steps
        done = self._obstacle_collision() or self._out_of_bounds() or terminated
        
        costs = self._ctrl_cost_weight * np.sum(np.square(action))
        reward = obs[0]/10 - costs - done + terminated
        
        if self.render_mode == "human":
            self.render()
        return obs, reward, done, terminated, {}


    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        super().reset(seed=seed)
        self.episode_step = 0
        return self.get_state()


    def reset_to(self, state):
        assert state.shape == (31,), "Need the full 31 states to reset qpos and qvel"
        self.reset()
        self.set_state(state[:16], state[16:]) # Mujoco method to set qpos, qvel
        self.episode_step = 0
        return self.get_state()


    def get_state(self):
        """Get reduced state (x, y, car_orientation) or full state (qpos, qvel)"""
        if self.reduced_model:
            x, y = self.data.qpos[:2].copy()
            body_rot = R.from_quat(self.data.qpos[3:7].copy(), scalar_first=True).as_euler("xyz", degrees=False)
            theta = body_rot[-1] # rotation wrt z
            vx, vy = self.data.qvel[:2].copy()
            theta_dot = self.data.qvel[5].copy()
            return np.array([x, y, theta, vx, vy, theta_dot])
           
        return np.concatenate((self.data.qpos.copy(), self.data.qvel.copy()))


    def reset_model(self):
        qpos = self.reset_qpos.copy()
        qpos[self._states_to_noise] += np.random.uniform(low=-self._reset_noise_scale,
                                                         high=self._reset_noise_scale,
                                                         size=len(self._states_to_noise))
        orientation = np.random.uniform(low=-self._reset_noise_scale,
                                        high=self._reset_noise_scale)
        qpos[3:7] = R.from_euler("z", orientation, degrees=False).as_quat(scalar_first=True)
        self.set_state(qpos, self.init_qvel.copy()) # no velocity at the start
        return self.get_state()
    
    
    def rollout(self, initial_state, actions):
        """Rollout a sequence of actions from a given initial state"""
        assert initial_state.shape == (31,), "Need the full 31 states to reset qpos and qvel"
        H = actions.shape[0]
        assert actions.shape == (H, self.action_size)
        
        Rewards = np.zeros(H)
        Traj = np.zeros((H+1, self.state_size))
        self.reset_to(initial_state)
        Traj[0] = self.get_state()
        
        for t in range(H):
            Traj[t+1], Rewards[t], done, _, _ = self.step(actions[t])
            if done: break

        return Traj[:, :t+2], Rewards[:t+1]
    

#%% Trajectory plotting


def plot_traj(env, Traj, title=None):

    fig, ax = nice_plot()
    plt.axis("equal")
    for i in range(env.num_obstacles):
        cylinder = pat.Circle(xy=env.obstacle_xy[i],
                              radius=env.obstacle_radius[i], color="red")
        ax.add_patch(cylinder)
       
    furthest_obstacle = np.argmax(env.obstacle_xy[:,0])
    furthest_obstacle = env.obstacle_xy[furthest_obstacle,0] + env.obstacle_radius[furthest_obstacle,0]
    x_min, x_max = min(Traj[:,0]), max(max(Traj[:,0]), furthest_obstacle)
    plt.plot([x_min, x_max], [env.xy_bounds[0, 1], env.xy_bounds[0, 1]], color="red")
    plt.plot([x_min, x_max], [env.xy_bounds[1, 1], env.xy_bounds[1, 1]], color="red")
    plt.plot(Traj[:,0], Traj[:,1], linewidth=3)
    plt.xlabel("x")
    plt.ylabel("y")
    if title is not None:
        plt.title(title)
    plt.show()


def traj_comparison(env, traj_1, label_1, traj_2, label_2, title="",
                    traj_3=None, label_3=None, traj_4=None, label_4=None,
                    legend_loc='best'):
    
    """Compares given quadcopter trajectories."""
    
    assert len(traj_1.shape) == 2, "Trajectory 1 must be a 2D array"
    assert len(traj_2.shape) == 2, "Trajectory 2 must be a 2D array"
    if traj_3 is not None:
        assert len(traj_3.shape) == 2, "Trajectory 3 must be a 2D array"
    if traj_4 is not None:
        assert len(traj_4.shape) == 2, "Trajectory 4 must be a 2D array"

    fig, ax = nice_plot()
    if title is not None:
        plt.title(title)
    plt.axis("equal")
    for i in range(env.num_obstacles):
        cylinder = pat.Circle(xy=env.obstacle_xy[i],
                              radius=env.obstacle_radius[i], color="red")
        ax.add_patch(cylinder)
        
    x_min = min(min(traj_1[:,0]), min(traj_2[:,0]))
    furthest_obstacle = np.argmax(env.obstacle_xy[:,0])
    furthest_obstacle = env.obstacle_xy[furthest_obstacle,0] + env.obstacle_radius[furthest_obstacle,0]
    x_max = max([max(traj_1[:,0]), max(traj_2[:,0]), furthest_obstacle])
    plt.plot([x_min, x_max], [env.xy_bounds[0, 1], env.xy_bounds[0, 1]], color="red")
    plt.plot([x_min, x_max], [env.xy_bounds[1, 1], env.xy_bounds[1, 1]], color="red")
    plt.plot(traj_1[:,0], traj_1[:,1], label=label_1, linewidth=3)
    plt.plot(traj_2[:,0], traj_2[:,1], label=label_2, linewidth=3)
    if traj_3 is not None:
        plt.plot(traj_3[:,0], traj_3[:,1], label=label_3, linewidth=3)
    if traj_4 is not None:
        plt.plot(traj_4[:,0], traj_4[:,1], label=label_4, linewidth=3)
    plt.legend(frameon=False, labelspacing=0.3, handletextpad=0.2, handlelength=0.9, loc=legend_loc)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def nice_plot():
    """Makes the plot nice"""
    fig = plt.gcf()
    ax = fig.gca()
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['font.sans-serif'] = ['Palatino Linotype']
    ax.spines['bottom'].set_color('w')
    ax.spines['top'].set_color('w') 
    ax.spines['right'].set_color('w')
    ax.spines['left'].set_color('w')
    
    return fig, ax



