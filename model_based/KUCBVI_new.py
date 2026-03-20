import cvxpy as cp
import numpy as np
import math


class GridWorld:
    def __init__(self,k,d,size=8,danger=[7,1],goal=[4,5],wall=[2,5],coins=[(1,6),(4,2),(5,5)], horizon=50):
        self.noise = 0.1
        self.k=k
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
        

    def get_feedback_and_features(self):
        weights = [0.1, 1.0, 2.0, 3.0]
        if (not self.sparse):
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
        return feedback,feedback_given,self._features()
    
    def true_return(self):
        weights = [0.1, 1.0, 2.0, 3.0]
        if(self.sparse==False):
            true_reward = 2*(1-(self.pos==self.danger))*(weights[self.collected]*(self.collected + 1.32*(self.pos==self.goal)) - 5*(self.t-14)/self.horizon + 5*36/50)
        else:
            true_reward = (1-(self.pos==self.danger))*(weights[self.collected]*(self.collected + 1.32*(self.pos==self.goal)))
        return true_reward
                
    
    def _features(self):
        """return trajectory features phi(tau)"""
        x, y = self.pos
        xg, yg = self.goal
        xd, yd = self.danger
        dist_to_goal = abs(x-xg) + abs(y-yg)
        dist_to_danger = abs(x-xd) + abs(y-yd)
        at_danger = int(self.pos == self.danger)
        at_goal = int(self.pos == self.goal and (at_danger==0))
        coin_indicator = [int(c in self.collected_coins) for c in self._init_coins]
        return np.array([dist_to_goal, dist_to_danger, at_goal, at_danger] + coin_indicator, dtype=float)
        

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
    


class RewardModel:
    def __init__(self, k,d,C=0.0):
        self.k = k
        self.d = d
        self.W = np.zeros((k,d))
        self.C = C
        
    def estimate_W(self, X, Y, reg=1e-3):
        n, d = X.shape
        W = cp.Variable((self.k, self.d))  # optimization variable, not vectorized, shape(k,d)
        loss = 0
        for i in range(n):
            phi = X[i]  
            yi = int(Y[i])
            logits = W @ phi 
            ## yi cannot be more than (k-1), so if assigning deterministic rewards without crafting W*, be careful
            loss += -(logits[yi] - cp.log_sum_exp(logits))
        
        loss = loss/n + reg*cp.norm(W, "fro")**2
        prob = cp.Problem(cp.Minimize(loss))
        prob.solve(solver=cp.MOSEK)

        self.W = W.value
        return self.W
    
    def reward_probabilities(self, phi):
        """estimate P(y|tau)"""
        logits = self.W @ phi
        logits-= np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits/np.sum(exp_logits)
    
    def reward_estimate(self,phi):
        """returns the average, expected reward given the reward probabilities"""
        return np.sum(np.array([i*self.reward_probabilities(phi)[i] for i in range(self.k)]))
    
    def optimistic_reward(self, phi, n_samples):
        """
        optimism term included
        """
        base = self.reward_estimate(phi)
        n = max(n_samples, 1)
        bonus = self.C / np.sqrt(n)
        optimistic = base + bonus
        return min(optimistic, self.k - 1)


###----------Traning loop-------------###
def train(N=20,m=50,k=6,eta=0.1,epsilon=0.1,grid_size=8,danger=[7,1],goal=[4,5], wall=[2,5] ,horizon=50,coins=None,seed=0,noise=0.1,sparse=False):
    np.random.seed(seed)
    queries = 0
    steps = 0
    if coins is None:
        coins=[(1,6),(4,2),(5,5)]
    print("hi")
    d = 4+len(coins)
#     W_true = generate_W_true(k,d)
    env = GridWorld(k,d,size=grid_size, danger=danger, goal=goal, wall=wall, coins=coins, horizon=horizon)
    env.noise = noise
    env.sparse = sparse
    R_max = env.R_max
    if sparse:
        R_max = env.R_max_sparse
        
    policy = Policy(grid_size=grid_size, action_dim=4)
    
    ## initialize weights w_0
    reward_model = RewardModel(k, d, C=10)
    reward_model.W = np.zeros((k,d))
    
    all_data_X, all_data_Y = [], []
    avg_true_rewards,avg_coins,avg_est_rewards,avg_labels = [], [], [],[]
    flag = 0
    for n in range(N):
        if flag: break
        print(n)
        avg_true_reward_this_iter = 0
        avg_est_reward_this_iter = 0
        if(n<100):
            G_range = 10
        elif (n>100 ):
            G_range = 140

        for g in range(G_range):
            steps+=1
            rollout_trajectories = []
            avg_true_reward_this_iter = 0
            avg_est_reward_this_iter = 0
            rewards = []
            labels = []
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
                __,y, phi = env.get_feedback_and_features()
                phi = np.array(phi, dtype=float)   
                rollout_trajectories.append((traj, phi, y))
                rewards.append(env.true_return())
                labels.append(__)
            if(np.mean(rewards)>R_max):
                flag = 1
                break

                
            ## now with these m rollouts, approximate the expectation of estimated reward under policy pi
            grad_theta = np.zeros_like(policy.theta)
            n_samples = max(len(all_data_X), 1)
            R_hats = [reward_model.optimistic_reward(phi, n_samples) for _, phi, _ in rollout_trajectories]
            b = float(np.mean(R_hats))  # baseline
                        
            for (traj,phi_tau,y), r_hat in zip(rollout_trajectories,R_hats):
                
                avg_est_reward_this_iter+=r_hat/len(rollout_trajectories)
#                 avg_true_reward_this_iter+=y/len(rollout_trajectories)
                temp = r_hat-b
                    
                for state,action in zip(traj["states"],traj["actions"]):
                    s_idx, grad_row = policy.grad_log_prob(state, action)
                    grad_theta[s_idx] += grad_row*(temp)

            grad_theta = grad_theta/len(rollout_trajectories)
            policy.theta += eta*grad_theta
            avg_true_reward_this_iter = np.mean(rewards)
            avg_label_this_iter = np.mean(labels)
            
        done = False
        s = env.reset()
        while not done:
            a,_ = policy.act(s)
            s,done = env.step(a)
        _,y,phi = env.get_feedback_and_features()
        all_data_X.append(phi)
        all_data_Y.append(y)
        reward_model.W = reward_model.estimate_W(np.array(all_data_X),np.array(all_data_Y), reg = 1e-3)
         
        traj,phi,y = rollout_trajectories[-1]

        coins_this_iter = traj["coins"]
        
        ## storing some info
        avg_labels.append(avg_label_this_iter)
        avg_est_rewards.append(avg_est_reward_this_iter)
        avg_true_rewards.append(avg_true_reward_this_iter)
        avg_coins.append(coins_this_iter)
        
        # update estimate of weight matrix W
        reward_model.W = reward_model.estimate_W(np.array(all_data_X),np.array(all_data_Y), reg = 1e-3)

        print(f"Iter {n:02d}: avg_label={avg_labels[-1]:.3f}, avg_true_reward={avg_true_rewards[-1]:.3f},coins_this_episode={avg_coins[-1]:.2f}")
    
    return policy, reward_model, avg_true_rewards, avg_est_rewards, reward_model.W,steps,queries


