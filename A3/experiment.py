# 
from typing import Any, Literal, Union

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt 

import sys 

class Settings(BaseModel):
    b : int = Field(default=1,description="branching factor")

    epsilon : float = Field(default=0.1, description="chance an exploratory action will be preformed")

    num_states : int = Field(default=1000,description="Total number of states")

    num_actions : int = Field(default=2,description="total actions that can be taken at a state")

    sample_task : int = Field(default = 40,description="number of episodic task preformed")

    max_steps : int = Field(default = 20000, description="how many expected updates to take")

    interval : int = Field(default = 200, description="every steps evaulate using a greedy policy")

    runs : int = Field(default = 1000, description="Number of runs for the policy to average over")


class EnviornmentModel(BaseModel): 
    config : Settings

    rewards : NDArray[Shape["* num_states, * num_actions, * b"], float]

    starting_state : int = Field(default = 0, description="episode always starts in this state")

    terminal_state : int = Field(default = 5, description="if this state is chosen the episode ends")

    state_action_branch_matrix : NDArray

    terminate_episode : bool = Field(default=False,description="Have we reached the terminal state")

    def _expected_reward(self,s,a,b):
        return self.rewards[s,a,b]
        
    def step(self,a,s):

        if np.random.rand() < 0.1:
            # print("BANGPOT")
            self.terminate_episode = True 

            return s,0,self.terminal_state,a
        branch_to_follow = np.random.choice(self.config.b)
        s_prime = self.state_action_branch_matrix[s,a,branch_to_follow]

        r = self._expected_reward(s_prime,a,branch_to_follow)

        if(s_prime == self.terminal_state):
            self.terminate_episode = True
            r=0
         # reward for transitioning to a state 

        return s,r,s_prime,a


class TrajectorySampling(BaseModel):
    config : Settings

    env : EnviornmentModel

    Q : NDArray[Shape["* num_states, 2"], float] = None

    def _resetQ(self):
        self.Q = np.zeros(shape=(self.config.num_states, self.config.num_actions))

    def greedy_epsilon(self,state):
        rand_val = np.random.rand()
        if rand_val > self.config.epsilon:
            # pick whatever a would maximixe this 
            return np.argmax(self.Q[state])
        else:
            return np.random.randint(2)
    
    def _update_rule(self,s,a):
        s_primes = self.env.state_action_branch_matrix[s,a]
        # rewards = np.array([self.env._expected_reward(s,a,b) for s in s_primes]) 
        rewards = self.env.rewards[s,a]
        max_q =  np.max(self.Q[s_primes], axis=1)
        return np.mean((0.9) * (max_q + rewards))

    def on_policy_update(self):
        # self._resetQ()

        hist = []
        s = self.env.starting_state
        for step in tqdm(range(self.config.max_steps)):
            
            action = self.greedy_epsilon(state = s)
            s,r,s_prime,a = self.env.step(s=s,a=action)
            
            self.Q[s,a] = self._update_rule(s,a)
            s = s_prime

            if self.env.terminate_episode == True:
                s =  self.env.starting_state
                self.env.terminate_episode = False 

            if step % self.config.interval == 0:
                reward_summary = self.compute_v()
                hist.append([step,reward_summary])

        # while(not self.env.terminate_episode):

        #     # hist.append((s,r,s_prime,a))

        #     # so in the other one if it reaches the end they just reset it 
        #     action = self.greedy_epsilon(state = s)
        #     s,r,s_prime,a = self.env.step(s=s,a=action)

        #     print(s,r,s_prime,a)

        #     self.Q[s,a] = self._update_rule(s,a)
        #     s = s_prime

            
            

        return hist 
        
    def uniform_update(self):
        # self._resetQ()
        hist = []
        for step in (range(self.config.max_steps)):
            s = step // self.config.num_actions % self.config.num_states
            a = step % self.config.num_actions
            self.Q[s,a] = self._update_rule(s,a)

            if (step % self.config.interval == 0):
                reward_summary = self.compute_v()
                hist.append([step,reward_summary])
                

        return hist

    def compute_v(self):
        returns = []

        for r in (range(self.config.runs)):
            total_reward = 0 
            s = self.env.starting_state
            self.env.terminate_episode = False 


            while(not self.env.terminate_episode):
                action = np.argmax(self.Q[s]) 
                s, r, s_prime, a = self.env.step(s=s, a=action)
                s = s_prime
                total_reward += r 
            

            returns.append(total_reward)
        return np.mean(returns)

def main():
    s = Settings()
    state_transition_rewards = np.random.normal(loc = 0, scale = 1, size = (s.num_states, s.num_actions, s.b))
    state_action_branch_matrix = np.random.randint(s.num_states, size=(s.num_states, s.num_actions, s.b))

    q = np.zeros(shape=(s.num_states, s.num_actions))
    
    env = EnviornmentModel(
        config=s,
        rewards = state_transition_rewards,
        terminal_state = np.random.randint(s.num_states),
        state_action_branch_matrix= state_action_branch_matrix
    )

    t = TrajectorySampling(
        config = s,
        env = env,
        Q=q
    )


    # (how do I average this over 200 runs. Do I need to make new trajectory classes and env classes)
    # r_h = t.on_policy_update()
    # r_h = np.array(r_h)

    # t.Q = np.zeros(shape=(s.num_states, s.num_actions))

    # r_h2 = t.uniform_update()
    # r_h2 = np.array(r_h2)

    # plt.plot(r_h[:,0],r_h[:,1])
    # plt.plot(r_h2[:,0],r_h2[:,1])
    # plt.ylabel('value of start state')
    # plt.xlabel('computation time, in expected updates')
    # plt.legend()
    # plt.show()

    # so its probably gonna be some thing like: 



    envs = [EnviornmentModel(
        config=s,
        rewards = np.random.normal(loc = 0, scale = 1, size = (s.num_states, s.num_actions, s.b)),
        terminal_state = np.random.randint(s.num_states),
        state_action_branch_matrix= np.random.randint(s.num_states, size=(s.num_states, s.num_actions, s.b))) for _ in range(s.sample_task)]
    
    all_rewards_uniform = []
    all_rewards_on_policy = []
    for env in trange(len(envs)):
        # we want to do this twice for uniform and on policy
        t_uniform = TrajectorySampling(
        config = s,
        env = envs[env],
        Q=np.zeros(shape=(s.num_states, s.num_actions))
        )

        t_on_policy = TrajectorySampling(
        config = s,
        env = envs[env],
        Q=np.zeros(shape=(s.num_states, s.num_actions))
        )
 
        rh_on_policy = t_on_policy.on_policy_update()
        all_rewards_on_policy.append(rh_on_policy)

        rh_uniform = t_uniform.uniform_update()
        all_rewards_uniform.append(rh_uniform)
        




    avg_return_on_policy = np.mean(np.array(all_rewards_on_policy)[:,:],axis=0)
    avg_return_uniform = np.mean(np.array(all_rewards_uniform)[:,:],axis=0)
    plt.plot(avg_return_on_policy[:,0],avg_return_on_policy[:,1],label='On-Policy')
    plt.plot(avg_return_uniform[:,0],avg_return_uniform[:,1],label = 'Uniform')
    plt.ylabel('value of start state')
    plt.xlabel('computation time, in expected updates')
    plt.title(f'{s.num_states} States averaged over {s.sample_task} runs')
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()

# Now we need to figure out how to plot everything up baby! 