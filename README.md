# Deep Reinforcement Learning

## Requirements
1. pybulletgym
2. gym
3. torch

## Description
We explore the decomposition of the Q value function (which acts as the critic) in actor-critic algorithms for a singular learning agent. We propose having multiple sub-critics that tackle two fundamental problems of reinforcement learning, reward shaping and sparse reward signals. Taking inspiration from multi-agent, hierarchical, and task-oriented reinforcement learning, we experiment with different Q value architectures to promote stable learning.

We apply our proposed Q value architectures on the algorithm, Twin Delayed Deep Deterministic Policy Gradient (TD3) and deploy the algorithm on a “reach” task with a two DOF manipulator.

As a proof of concept, we first experimented with the architecture, TD3 Decomposed (TD4), in Figure 1b. This approach was for the verification that a critic network with a decoupled state space would be able to solve the task. Figure 1d shows the architecture for Multiple Objective TD4 (moTD4), which gives each sub-critic network it's own objective function. This draws from the idea of HRL where the problem is broken down into separate tasks. We designed two sub-tasks for the “reach” task, one to reach in the x axis and one for the y axis. Figure 1c is the architecture for Q Summation TD4 (sumTD4). This network is inspired by MARL where each joint of the manipulator is treated as an individual agent. Each agent has its own sub-critic and their sum would represent the value estimated based on a singular reward signal. This technique required the addition of actors for each sub- critic.


## To train:
1. python decomp.py/mo_decomp.py/sum_comp.py (for TD4/moTD4/sumTD4)

## To visualize:
Use reacher_render.py with the necessary args




