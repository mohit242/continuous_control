[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

## Project 2: Continuous Control

### Problem statement

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that 
the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target 
location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of 
the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the
 action vector should be a number between -1 and 1.

For this implementation the second version of the unity environment is being used.
#### Goal
In particular, the agents must get an average score of +30 (over 100 consecutive episodes, and over
  all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each 
agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

**Environment description**
```
Number of Visual Observations (per agent): 0
Vector Observation space type: continuous
Vector Observation space size (per agent): 33
Number of stacked Vector Observation: 1
Vector Action space type: continuous
Vector Action space size (per agent): 4

```

### Learning Algorithm

