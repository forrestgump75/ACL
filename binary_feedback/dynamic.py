import cvxpy as cp
import numpy as np
import math

class GridWorld:
    def __init__(self,d,size=8,danger=[3,3],goal=[0,6],coins=[(1,1),(1,3),(6,2)], horizon=50):
        self.confidence = 0.9
        self.d=d
        self.size = size
        self.horizon = horizon
        self.goal = tuple(goal)
        self.danger = tuple(danger)
        self._init_coins = tuple(map(tuple, coins)) 
        self.coins = set(self._init_coins)
        self.collected_coins = set()
        self.done = 0
        self.threshold = 0
    
    def reset(self):
        self.done = 0
        self.pos = (6,0)
        self.t = 0
        self.collected = 0
        self.collected_coins = set()
        self.coins = set(self._init_coins)
        return self.pos
    
    def step(self, intended_action):
        probs = np.full(4, 0.03)
        probs[intended_action] = 0.91
        action = np.random.choice(4, p=probs)
        x, y = self.pos
        if action == 0: x = max(0, x-1)       # up
        if action == 1: x = min(self.size-1, x+1) # down
        if action == 2: y = max(0, y-1)       # left
        if action == 3: y = min(self.size-1, y+1) # right
        self.pos = (x,y)
        if self.pos in self.coins:
            self.collected += 1
            self.collected_coins.add(self.pos)
            self.coins.remove(self.pos)
        self.t += 1
        self.done = ((self.t >= self.horizon)or (self.pos==self.goal) or (self.pos==self.danger))
        return self.pos, self.done

    def true_return(self):
        weights = [0.1, 1.0, 2.0, 3.0]
        true_reward = (1-(self.pos==self.danger))*(weights[self.collected]*(self.collected + 1.32*(self.pos==self.goal)) - 5*(self.t-14)/self.horizon + 5*36/50)
        return 2*true_reward
    
    def get_feedback(self):
        return int(self.true_return()>=self.threshold)        
        

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)

class Policy:
    def __init__(self, grid_size, action_dim):
        self.grid_size = grid_size
        self.state_dim = grid_size * grid_size
        self.action_dim = action_dim
        self.theta = np.ones((self.state_dim, self.action_dim))
    
    def state_index(self, state):
        return state[0] * self.grid_size + state[1]
    
    def act(self, state):
        s_idx = self.state_index(state)
        probs = softmax(self.theta[s_idx])
        action = np.random.choice(len(probs), p=probs)
        return action, probs
    
    def grad_log_prob(self, state, action):
        """Return (state_index, grad_row) with grad_row shape (action_dim,)
           grad_row[j] = 1{j==action} - pi(j|s)"""
        s_idx = self.state_index(state)
        probs = softmax(self.theta[s_idx])
        grad_row = -probs.copy()
        grad_row[action] += 1.0
        return s_idx, grad_row
    


###----------Traning loop-------------###
def train(n=50, num_traj=100,m=500,eta=0.1,epsilon=0.1, alpha=0.75 ,grid_size=8,danger=[3,3],goal=[0,6],horizon=50,coins=None,seed=0):
    np.random.seed(seed)
    step_count = 0
    if coins is None:
        coins=[(1,1),(1,3),(6,2)]
    d = 4+len(coins)
    env = GridWorld(d,size=grid_size, danger=danger, goal=goal, coins=coins, horizon=horizon)
    policy = Policy(grid_size=grid_size, action_dim=4)
    
    ## initialise q_hat
    temp = []
    for _ in range(num_traj):
        s = env.reset()
        done = False
        while not done:
            a, _ = policy.act(s)
            s, done = env.step(a)
        temp.append(env.true_return())
        
    ## compute alpha quantile of returns in temp
    env.threshold = 2*np.mean(temp) ## q_hat initialzed!
    
    flag = 0

    ## try to improve policy for 10 iterations--> k corresponding to policy learnt in kth iteration
    for h in range(n):
        if(flag): break
#         print(f"threshold: {env.threshold}")
#         print(f"itration number: {h}")
        for g in range(1000):
            step_count += 1
#             print(g)
            returns = []
            labels = []
            rollout_trajectories = []
            for i in range(m): ## sample trajectories under current policy pi to approiximate the theoretical expectation
                s = env.reset()
                traj = {"states": [], "actions": [], "steps":0, "coins":0}
                done = False

                while not done:
                    a, _ = policy.act(s)
                    traj["states"].append(s)
                    traj["actions"].append(a)
                    s, done = env.step(a)

                traj["steps"] = env.horizon
                traj["coins"] = env.collected
                y = env.get_feedback()

                returns.append(env.true_return())
                labels.append(y)
                rollout_trajectories.append((traj, y))
#             print(f"average reward: {np.mean(returns)}")
#             print(f"average feedback: {np.mean(labels)}")
            if(np.mean(returns)>31):        
                flag = 1
                break
            if(np.mean(labels)>0.95 and h<n-1): break
            
            
                ## now with these m rollouts, approximate the expectation of estimated reward under policy pi
            grad_theta = np.zeros_like(policy.theta)
            R_hats = [y for _, y in rollout_trajectories]
            b = float(np.mean(R_hats))  # baseline

            for (traj,y), r_hat in zip(rollout_trajectories,R_hats):

                temp = r_hat-b

                for state,action in zip(traj["states"],traj["actions"]):
                    s_idx, grad_row = policy.grad_log_prob(state, action)
                    grad_theta[s_idx] += grad_row*(temp)

            grad_theta = grad_theta/len(rollout_trajectories)
            policy.theta += eta*grad_theta
        
#         now sample multiple trajectories to update q_hat
        temp = []
        for _ in range(num_traj):
            s = env.reset()
            done = False
            while not done:
                a, _ = policy.act(s)
                s, done = env.step(a)

            temp.append(env.true_return())
        step_count += num_traj

        env.threshold = np.quantile(temp,alpha)
            
                      

    return policy,step_count

