from typing import Any, Literal, Union, List

import numpy as np
import matplotlib.pyplot as plt 
import pickle
from tqdm import trange
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field


from racetrack import EnviornmentA,State,RaceTrack

class EpsilonSoftPolicy(BaseModel):
    epsilon: float = Field(default=0.2, description="Defines probability we preform an exploratory action") 
    num_actions : int 
    target_policy : NDArray

    def policy(self,S):
        rand_val = np.random.rand()
        # greedy action 
        if rand_val > self.epsilon:
            return self.target_policy[S], (1 - self.epsilon + self.epsilon / self.num_actions)

        else:
            a = np.random.choice(self.num_actions)
            if (a == self.target_policy[S]):
                return a, (1 - self.epsilon + self.epsilon / self.num_actions)
            else:
                return a , (self.epsilon / self.num_actions)

# random seed should go here??
class OffPolicyMC(BaseModel):
    total_epsiodes : int
    l : float # lambda
    Q : NDArray # num of states == width x height of track 
    C : NDArray
    target : NDArray
    env : EnviornmentA
    behavior: EpsilonSoftPolicy = Field(default=None,description="behavioral policy")
    

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.behavior = EpsilonSoftPolicy(l = 0.9,
                                          num_actions=9,
                                          target_policy=self.target)

    def state_index(self, state: State):
        # Both velocity components are between 0 and 4. So [0,1,2,3,4] [0,1,2,3,4]
        # We have track_width*track_hegiht potential spaces the car can be in. (Including the ones that are zero) (most won't be populated).
        x,y = state.position
        vx,vy = state.velocity
        return y * self.env.track.shape[1] * 25 + x * 25 + vy * 5 + vx # not sure about this 

    def _update_G(self,G,r):
        return self.l*G + r

    def _update_C(self,C,W):
        return C + W # problem 1: Sliding in ALL OF C. SHOULD ONLY BE ONE

    def _update_Q(self,Q,W,C,G):
        return Q + (W/C)*(G - Q)  # think I'm also updating all of Q. 

    def _update_W(self,W,prob):
        return W*(1/prob)

    def episode(self,b,noise=True):
        terminate = False 
        state = self.env._reset_pos() 
    
        action, p = b(((*state.position, *state.velocity)))
        total_reward = 0
        hist = []

        while not terminate:
            observation, reward, terminate = self.env.step(a = action,state = state)
            # updates
            total_reward += reward
            state = observation
            action, p = b((*state.position, *state.velocity)) # hmmmmmm
            
            hist.append((state,action,reward,p)) 

            # start debugging this 
            # print("Step Summary",(state,action,reward,p))
            
        # breakpoint()
        return hist,total_reward

    def off_policy_mc_control(self):
        reward_hist = []
        for _ in trange(self.total_epsiodes, desc="Episodes"):
            b = self.behavior.policy
            t,total_reward = self.episode(b,True)
            G = 0
            W = 1
            reward_hist.append(total_reward)
            while t:
                state,action,reward,p = t.pop()
                G = self.l*G + reward
                S = (*state.position, *state.velocity)
                self.C[S][action] = self.C[S][action] + W
                self.Q[S][action] = self.Q[S][action] + (W/self.C[S][action]) * G - self.Q[S][action]
                # choose a new state based on argmaxing something
                self.target[S] = np.argmax(self.Q[S]) # actions will be stored (or the number that corresponds to an action)
                if action != self.target[S] : break
                W = W * (1 / p)
        return self.target,reward_hist

def main():
    train = False

    rt = RaceTrack(trackA=np.ones((32,17), dtype=float),trackB=np.ones((30,32), dtype=float))
    track = rt.generate_track_b()
    env = EnviornmentA(track=track)
    

    nA = 9
    nS =  (env.track.shape[0], env.track.shape[1], 5, 5) # now this is a tuple with 4 elements 
    total_ep = 100000
    l = 0.9 # lambda (discount)
    Q = np.random.normal(size=(*nS, nA))
    Q-=500
    C = np.zeros_like(Q)
    pi = np.argmax(Q, axis=-1)

    mc = OffPolicyMC(total_epsiodes=total_ep,
                    l = l,
                    Q = Q,
                    C = C,
                    target= pi,
                    env = env)
    if train:    
        target_policy,reward_hist = mc.off_policy_mc_control()
        filename = '100k_mc_resultsB_OP.pkl'

        # Use pickle to save the results
        with open(filename, 'wb') as file:
            pickle.dump((target_policy, reward_hist), file)

        print("DONE TRAINING")
    else:

        np.set_printoptions(threshold = np.inf)
        filename = '100k_mc_resultsB_OP.pkl'
        with open(filename, 'rb') as file:
            target_policy, n_reward_hist = pickle.load(file)

        

        print(target_policy)
        # print(n_reward_hist)
        plt.figure()
        plt.plot(range(1, total_ep + 1), n_reward_hist)  # Start range from 1 to avoid log(0)
        plt.xscale('log')
        plt.show()

        # we don't care about most of this we just want to run one episode
    for i in range(4):
        terminate = False 
        state = env._reset_pos() 
        action = target_policy[(*state.position,*state.velocity)]
        print("-----------(NEW EP)-----------")

        while not terminate:
            track[state.position[0], state.position[1]] = 0.6 
            observation, reward, terminate = env.step(a = action,state = state,noise=False)
            state = observation
            action = target_policy[(*state.position,*state.velocity)]
            print(state,action)
        
        ax = plt.subplot(2, 2,i+1)
        ax.axis('off')
        ax.imshow(track, cmap='Pastel1_r')

    plt.show()
           
            




True



    

if __name__=="__main__":
    main()

# probs (so state is only 2 things??)



