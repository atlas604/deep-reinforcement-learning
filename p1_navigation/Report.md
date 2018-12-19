# Project 1: Navigation - Report

### Learning Algorithm

A Deep Q-Network (DQN) is applied to solve the environment.  We replicate the DQN algorithm from the [LunarLander](https://github.com/atlas604/deep-reinforcement-learning/tree/master/dqn) environment as it has shown promising results.  

A Deep Q-network allows the agent to sample the environment by performing actions and storing the experience in memory.  The goal of the agent is to interact with the environment and select actions that maximize rewards.  

The implementation produces a Q-value for every possible action in a single forward pass and uses the vector to take an action (stochastically or choosing the one with the maximum value).  

We use a neural network with an input layer size of 37 which is equivalent to the action space of the environment and an output layer size of 4 which is equivalent to the number of actions the agent is able to perform (Up, Down, Left Right).  

Using this algorithm, the agent was able to complete the environment in less than 500 episodes.  

The same hyperparameters from the LunarLander exercise are also used in this implementation.  The episilon value is maximized in the start and gradually decreased as the agent learned and performed better.  This seemed pretty intuitive from a learning perspective to keep a reasonable amount of stochasticity to balance the agent’s learning behavior. As the agent becomes more and more knowledgeable, we still want it to explore different actions at a certain probability to possibly improve the policy.  


### Ideas for Future Work
