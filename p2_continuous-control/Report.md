# Project 2: Continuous Control - Report

### Learning Algorithm

The learning algorithm implemented is a Deep Deterministic Policy Gradient (DDPG) model.   Specifically, a modified [ddpg-pendulum](https://github.com/atlas604/deep-reinforcement-learning/tree/master/ddpg-pendulum) model is used to solve the environment.   

Unlike the Deep Q Network (DQN) algorithms, DDPG isn’t constrained with discrete and low-dimensional action spaces.  This is critical for our purposes of interacting with continuous action spaces.  
In DDPG, two neural networks are used instead of one, an actor and a critic.  The actor is used to approximate the optimal policy deterministically, which is essentially what it perceives as the best possible action.  The critic on the other hand uses the actor’s decision to evaluate the optimal action value function.

My first attempt was to use mostly default parameters to see how the model interacts with the [version 1] environment.  

![Attempt 1](./img/attempt%201.png)

As you can see, the model had performed extremely slow and poorly while having extremely high variance.  Perhaps it could learn to solve the environment with more episodes, but this was way too inefficient.  

After numerous attempts at tuning and testing the model, I found significant improvements by normalizing the batch sizes for both the actor and critic neural network and increased the learning rate.  

The DDPG code was modified to gather the data from multiple agents simultaneously and add their experience to the replay buffer.  

![Attempt 2](./img/attempt%202.png)

Hyperparameters chosen:

```BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0000   # L2 weight decay
```

Both Actor and Critic neural networks were initialized with two hidden layers of 128 nodes and with batch normalization implemented between each linear functions.  

The current modified DDPG model was able to be solve the [version 2] environment within 124 episodes.   


### Ideas for Future Work

The DDPG hyperparameters can definitely be optimized to improve the model.  With optimal values the agent can most likely be trained in under 100 episodes.  

I have some concerns about training beyond an average reward of +30.  The model might be unable to stabilize after a high number of episodes and experience significant diminishing returns or simply perform worse.  Over-training the model could introduce lots of bias but it’ll be interesting to see what will happen after training the agent after an indefinite amount of episodes.  

I’ve also seen other implementations utilizing a Proximal Policy Optimization (PPO) algorithm which yielded a more consistently learning curve.  I’m eager to try to do the same implementation to understand how it compares to DDPG and perhaps other algorithms as well.  
