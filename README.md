# Deep Q-Network (DQN) Lunar Lander Project

## Key Takeaways	 
This project uses a Deep Q-Network (DQN) to teach an agent how to land a spacecraft in OpenAI Gym’s Lunar Lander environment. After training for 604 episodes, the agent averaged 201.93 points over 100 test runs, demonstrating core DQN ideas: experience replay, a separate target network, and epsilon-greedy action selection.

## 1. Understanding the Lunar Lander
### 1.1 Overview  
Lunar Lander is a simulation where you control a rocket trying to land gently on a flat pad. It’s part of the Box2D physics suite and models a trajectory optimization problem.

### 1.2 State Space  
Each step, the agent sees 8 numbers:  
- x and y position  
- x and y velocity  
- angle and angular velocity  
- two bits indicating if each leg touches the ground  

### 1.3 Action Space  
The agent picks one of four actions:  
0. Do nothing  
1. Fire left engine  
2. Fire main engine  
3. Fire right engine  

### 1.4 Rewards  
- Soft landing: +100 to +140  
- Leg contact: +10 per leg  
- Crash: –100  
- Main engine: –0.3 per frame  
- Side engines: –0.03 per frame  
- Moving away: small penalty  
Goal: average ≥ 200 points over 100 episodes.

## 2. Deep Q-Network Fundamentals  

### 2.1 Q-Learning Basis  
DQN builds on Q-learning, which learns Q*(s,a), the expected total reward if you take action a in state s. It uses the Bellman equation:  

Q*(s,a) = E[r + γ max Q*(s',a') | s,a]


### 2.2 Neural Network Approximation  
Instead of a huge Q-table, DQN uses a neural network $$Q(s,a;\theta)$$ to estimate Q-values for continuous state spaces.

### 2.3 Network Structure  
- Input: 8 neurons  
- Hidden 1: 64 ReLU neurons  
- Hidden 2: 64 ReLU neurons  
- Output: 4 linear neurons  
≈ 4,800 parameters

## 3. Key Techniques for Stability  

### 3.1 Experience Replay  
Store up to 100,000 past experiences $$(s,a,r,s',\text{done})$$ in a buffer. Each update samples a random batch of 64, breaking correlations between consecutive frames and improving learning stability.

### 3.2 Target Network  
Maintain two networks:  
- Main network $$\theta$$ for selecting actions  
- Target network $$\theta^-$$ for computing stable targets  
After each training step, softly update $$\theta^-\leftarrow\tau\theta + (1-\tau)\theta^-$$ with $$\tau=0.001$$.

### 3.3 Epsilon-Greedy Exploration  
Start with $$\varepsilon=1.0$$ (fully random), decay by 0.995 per episode down to 0.01. With probability $$\varepsilon$$, choose a random action; otherwise pick $$\arg\max_a Q(s,a)$$.

## 4. Training Details  

### 4.1 Loss Function  
Minimize the mean squared error between predicted Q-values and targets  

L(θ) = E[(y - Q(s,a;θ))²]

where y = r + γ max Q(s',a';θ⁻)


### 4.2 Training Loop  
1. Initialize both networks randomly and copy weights to target.  
2. For each episode up to 2000:  
   - Reset environment.  
   - At each timestep (up to 1000):  
     - Choose action by epsilon-greedy.  
     - Step environment, collect $$(s,a,r,s',\text{done})$$.  
     - Append to replay buffer.  
     - Every 4 steps, if buffer ≥ 64 samples:  
       - Sample batch, compute targets, update main network, then soft-update target.  
     - End episode on done.  
   - Decay $$\varepsilon$$.  
   - Check if average of last 100 episodes ≥ 200; if so, stop.


## 5. Results & Performance  

| Episodes | Avg. Score | Stage                  |
|----------|------------|------------------------|
| 100      | –148.20    | Initial exploration    |
| 200      | –78.12     | Basic learning         |
| 300      | –41.77     | Skill development      |
| 400      | +46.20     | Early successes        |
| 500      | +163.13    | Near target            |
| **604**  | **+201.93**| Environment solved     |

- Converged in 604 episodes (~31.6 min)  
- Improvement: ~0.58 points/episode  
- Exceeded 200-point goal  

## 6. Hyperparameters  

| Parameter         | Value    |
|-------------------|----------|
| Replay Buffer     | 100,000  |
| Discount Factor γ | 0.995    |
| Learning Rate α   | 0.001    |
| Soft Update τ     | 0.001    |
| Batch Size        | 64       |
| Update Every      | 4 steps  |
| ε-Decay           | 0.995    |
| ε-Min             | 0.01     |

## 7. Theory  

- **Bellman Optimality**: Recursive relation for $$Q^*$$.  
- **Temporal Difference**: Updates Q(s,a) using $$r + \gamma\max Q(s',a')$$.  
- **Function Approx.**: Neural nets generalize Q-values over continuous states.


## 8. Performance & Scalability  

- **@tf.function**: Speeds up training loops  
- **Vectorized Ops**: Efficient minibatch updates  
- **Circular Buffer**: Controls memory usage  

## 9. Limitations & Future Work  

- Only tested on Lunar Lander  
- Fixed network size; no adaptive architecture  
- Uniform sampling; could use prioritized replay  
- Manual hyperparameter tuning  

## 10. Conclusion  
This DQN solution shows how combining Q-learning with deep neural networks solves complex control tasks. The agent reliably lands the Lunar Lander, hitting an average score above 200 in under 600 episodes. The code is modular, well-tested, and ready for extensions like Double DQN or Prioritized Replay—offering a solid base for further research in reinforcement learning.

