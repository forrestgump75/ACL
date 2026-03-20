from model import *

if __name__=="__main__":
    
    avg_true_vector = []
    
    for k in range(2,32):
        trained_policy, trained_reward_model, avg_true, avg_est = train(N=300,m=50,k,eta=0.05,epsilon=1e-2,grid_size=8,danger=[3,3],goal=[0,6],horizon=50,coins=[(1,1),(1,3),(6,2)],seed=k)
        avg_true_vector.append(avg_true)
 
        
