import cvxpy as cp
import numpy as np
import math
import random

class GridWorld:
    def __init__(self,k,d,size=8,danger=[7,1],goal=[4,5],wall=[2,5],coins=[(1,6),(4,2),(5,5)], horizon=50):
        self.noise = None
        self.k = k
        self.d=d
        self.size = size
        self.horizon = horizon
        self.goal = tuple(goal)
        self.danger = tuple(danger)
        self.wall = tuple(wall)
        self._init_coins = tuple(map(tuple, coins)) 
        self.coins = set(self._init_coins)
        self.collected_coins = set()
        self.done = 0
        self.threshold = 0
        self.sparse = False
        self.R_max_sparse = 12.96
        self.R_max = 32.92
    
    def reset(self):
        self.done = 0
        self.pos = (0,7)
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
        if((x,y)!=(self.wall)):
            self.pos = (x,y)
        if self.pos in self.coins:
            self.collected += 1
            self.collected_coins.add(self.pos)
            self.coins.remove(self.pos)
        self.t += 1
        self.done = ((self.t >= self.horizon)or (self.pos==self.goal) or (self.pos==self.danger))
        return self.pos, self.done

    def get_feedback(self):
        weights = [0.1, 1.0, 2.0, 3.0]
        if(self.sparse==False):
            true_reward = 2*(1-(self.pos==self.danger))*(weights[self.collected]*(self.collected + 1.32*(self.pos==self.goal)) - 5*(self.t-14)/self.horizon + 5*36/50)
        else:
            true_reward = (1-(self.pos==self.danger))*(weights[self.collected]*(self.collected + 1.32*(self.pos==self.goal)))
        ## now we quantize it into k bins
        R = self.R_max_sparse if self.sparse else self.R_max
        edges = np.linspace(0,R,self.k+1)
        feedback = self.k-1
        for i in range(len(edges)-1):
            if(edges[i]<=true_reward and true_reward<edges[i+1]):
                feedback = i
                break
                
        # label noise
        probs = [0.0] * self.k
        probs[feedback] = 1-self.noise+self.noise/self.k
        rem = 1-probs[feedback]
        rem_distributed = rem / (self.k - 1)
        for i in range(self.k):
            if probs[i] == 0.0:
                probs[i] = rem_distributed
   
        feedback_list = [i for i in range(self.k)]
        feedback_given = np.random.choice(feedback_list,p=probs)
        return feedback_given
    
    def true_return(self):
        weights = [0.1, 1.0, 2.0, 3.0]
        
        if(self.sparse==False):
            true_reward = 2*(1-(self.pos==self.danger))*(weights[self.collected]*(self.collected + 1.32*(self.pos==self.goal)) - 5*(self.t-14)/self.horizon + 5*36/50)
        else:
            true_reward = (1-(self.pos==self.danger))*(weights[self.collected]*(self.collected + 1.32*(self.pos==self.goal)))
        return true_reward
        

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
def train(m=50,k=2,eta=0.5,epsilon=0.1,grid_size=8,danger=[7,1],goal=[4,5],wall=[2,5],horizon=50,coins=None,seed=0,noise=0.1,sparse=False):
    np.random.seed(seed)
    queries = 0
    steps = 0
    if coins is None:
        coins=[(1,6),(4,2),(5,5)]
    d = 4+len(coins)
    env = GridWorld(k,d,size=grid_size, danger=danger, goal=goal, wall=wall, coins=coins, horizon=horizon)
    policy = Policy(grid_size=grid_size, action_dim=4)
    R_max = 31
    if(sparse): R_max = 12.5
    env.noise = noise
    env.sparse = sparse
    flag = 0
    average_returns = []
    for h in range(1):
        if(flag): break
        for g in range(15000):
            steps+=1
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
                queries+=1
                returns.append(env.true_return())
                labels.append(y)
                rollout_trajectories.append((traj, y))
            if(g%1==0):
                print(g)
                print(f"average reward: {np.mean(returns)}")
                print(f"average feedback: {np.mean(labels)}")
            average_returns.append(np.mean(returns))
            if(np.mean(returns)>R_max):
                flag = 1
                break            
            
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


                      

    return policy,queries,steps,average_returns
