# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 09:38:57 2025

@author: Jean-Baptiste Bouvier

Scripted policy for the Car navigation task
"""

import time
import numpy as np
from car import CarEnv, plot_traj
from scipy.spatial.transform import Rotation as R


class ScriptedPolicy():
    """ Scripted policy to make the Waymo car slalom between the two obstacles"""
    def __init__(self, verbose=False):
        self.verbose = verbose       
        
    def reset(self):
        """Resets the waypoints tracking of the policy"""
        self.waypoint_id = 0
        
        
    def evaluate(self, obs):
        """Evaluate the scripted policy"""
        assert hasattr(self, 'waypoint_id'), "The policy needs to be reset at the beggining of each run"
        
        x, y = obs[:2]
        body_rot = R.from_quat(obs[3:7].copy(), scalar_first=True).as_euler("xyz", degrees=False)
        theta = body_rot[-1] # rotation wrt z
        
        if self.waypoint_id == 0:
            action = np.array([0.2, 0.5]) # go up above first cylinder
            if theta > 0 and y > 0.15 and x > 0.7:
                self.waypoint_id = 1
                if self.verbose:
                    print(f"transition to waypoint 1 at x = {x:.1f}  y = {y:.1f}")
                
        elif self.waypoint_id == 1:
            action = np.array([-theta, 0.4]) # correct course not to go too high in y
            if x > 2.4 or (x > 1.9 and y > 0.5): # cleared first obstacle
                self.waypoint_id = 2
                if self.verbose:
                    print(f"transition to waypoint 2 at x = {x:.1f}  y = {y:.1f}")
                
        elif self.waypoint_id == 2:
            action = np.array([-1., 0.4]) # go all the way down to avoid second cylinder
            if y < 0:
                self.waypoint_id = 3
                if self.verbose:
                    print(f"transition to waypoint 3 at x = {x:.1f}  y = {y:.1f}")
                
        elif self.waypoint_id == 3:
            action = np.array([1., 0.5]) # correct course going back up
            if x > 5:
                self.waypoint_id = 4
                if self.verbose:
                    print(f"transition to waypoint 4 at x = {x:.1f}  y = {y:.1f}")
        
        elif self.waypoint_id == 4:
            action = np.array([-y*abs(y), 1.]) # go fast and back to the middle
            
        return action
    
    
    def rollout(self, env: CarEnv, seed=None):
        """Rolls out an entire trajectory of the scripted policy"""
        Traj = np.zeros((env.max_episode_steps, env.state_size))
        Actions = np.zeros((env.max_episode_steps, env.action_size))
        Rewards = np.zeros((env.max_episode_steps, 1))
        Traj[0] = env.reset(seed=seed)
        self.reset()
        
        for t in range(env.max_episode_steps-1):
            Actions[t] = self.evaluate(Traj[t])
            Traj[t+1], Rewards[t], done, _, _ = env.step(Actions[t])
            if done: break
        
        Actions[t+1] = self.evaluate(Traj[t+1]) # to have the same number of actions as states
        print(f"Seed {seed}:   total reward: {sum(Rewards[:,0]):.1f}   final x: {Traj[t+1,0]:.2f}")
        env.close()
        return Traj[:t+2], Actions[:t+2], Rewards[:t+1]
            
        
        
        
#%% testing

if __name__ == "__main__":
    
    seed = 10
    env = CarEnv(render_mode="human", reduced_model=False, verbose=True)
    policy = ScriptedPolicy(verbose=True)
    Traj = np.zeros((env.max_episode_steps+1, env.state_size))
    Traj[0] = env.reset(seed=seed)
    policy.reset()
    reward = 0
    
    for t in range(env.max_episode_steps):
        
        action = policy.evaluate(Traj[t])
        Traj[t+1], r, done, _, _ = env.step(action)
        reward += r
        if env.render_mode == "human":
            env.render()
            time.sleep(0.05)
        if done: 
            break
    
    print(f"Total reward: {reward:.1f}   final x-position: {Traj[t+1,0]:.2f}")
    env.close()
    plot_traj(env, Traj[:t+2])


#%% Evaluation

    env = CarEnv(reduced_model=False, verbose=True)
    policy = ScriptedPolicy(verbose=False)
    
    for seed in range(1000):
        Traj, Actions, Rewards = policy.rollout(env, seed)
        if Traj.shape[0] < env.max_episode_steps:
            print("Failure")
            plot_traj(env, Traj[:t+2])
            break
        