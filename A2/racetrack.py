# Generate a RaceTrack
from typing import Any, Literal, Union, List, NamedTuple

import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings
import itertools

import matplotlib.pyplot as plt

START= 0.8 
FINISH= 0.4 

# Do not make a policy that applies for both race tracks 
class RaceTrack(BaseModel):

    trackA : NDArray[Shape["32, 17"],float]
    trackB : NDArray[Shape["30, 32"],float]

    def generate_track_a(self):
        self.trackA[14:, 0] =   0  #[14 and onwards in the first column]
        self.trackA[21:,1]  =   0
        self.trackA[29:,2]  =   0
        self.trackA[7:,9] =   0
        self.trackA[6:,10:17] =   0
        self.trackA[0:3,0] = 0
        self.trackA[0:2,1] = 0
        self.trackA[0:1,2] = 0
        self.trackA[0,3] = 0

        # Distinguish between Start and Finish
        self.trackA[31,3:9] = START
        self.trackA[0:6,16] = FINISH

        return self.trackA
    def generate_track_b(self):
        for i in range(14):
            self.trackB[:(-3 - i), i] = 0
        
        self.trackB[3:7, 11] = 1
        self.trackB[2:8, 12] = 1
        self.trackB[1:9, 13] = 1
    
        self.trackB[0, 14:16] = 0
        self.trackB[-17:, -9:] = 0
        self.trackB[12, -8:] = 0
        self.trackB[11, -6:] = 0
        self.trackB[10, -5:] = 0
        self.trackB[9, -2:] = 0

        self.trackB[-1] = np.where(self.trackB[-1] == 0, 0, START)
        self.trackB[:, -1] = np.where(self.trackB[:, -1] == 0, 0, FINISH)
        return self.trackB

class State(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray

class EnviornmentA(BaseModel):
    track : Union[NDArray[Shape["32, 17"], float],NDArray[Shape["30, 32"], float]]

    actions : NDArray = Field(default_factory=lambda: np.array(list(itertools.product([-1, 0, 1], [-1, 0, 1]))), description="All 9 actions the car can take")
    starting_states: NDArray = Field(default_factory=lambda: np.array([]), description="Starting squares of the car")
    ending_states: NDArray = Field(default_factory=lambda: np.array([]), description="Ending squares of the car")
    noise: float = Field(default=0.1, description="Probability that we do not update our speed")
    
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.starting_states = np.dstack(np.where(self.track == START))[0]
        self.ending_states = np.dstack(np.where(self.track == FINISH))[0]
    
    def _vel_check(self,curr_vel,accel):
        # next_vel = np.clip(curr_vel + accel, a_min=0, a_max=4)


        temp_y_acc, temp_x_acc = curr_vel + accel

        if temp_y_acc < -4: temp_y_acc = -4
        if temp_y_acc > 0: temp_y_acc = 0 
        if temp_x_acc < 0: temp_x_acc = 0
        if temp_x_acc > 4: temp_x_acc = 4

        next_vel = np.array([temp_y_acc,temp_x_acc])
        if next_vel[0]==0 and next_vel[1]==0:
            next_vel = curr_vel
            return next_vel
        
        return next_vel
    

    def _finish_check(self,proj_pos):

        if(self.track[proj_pos[0],proj_pos[1]] == FINISH):
            # print("FINISHED BOIIII")
            return True
        
        return False 
    
    # check if the vehicle hit the wall. If it does then reset 
    def _accident_check(self,proj_pos):
        proj_y = proj_pos[0]
        proj_x = proj_pos[1] 
        H,W = self.track.shape
        # print(proj_y,proj_x)

        if (proj_y >= H or proj_y < 0) or (proj_x < 0 or proj_x >= W): 
            return True
        if(self.track[proj_y,proj_x] == 0.0):
            return True

    
    def _reset_pos(self):
        # choose something on the random starting state
        start_idx = np.random.choice(self.starting_states.shape[0])
        return State(self.starting_states[start_idx],np.array([0,0]))

    def step(self,a,state: State):
        # need to get the current state
        
        reward = -1
        terminate = False
        position, velocity = state

        if np.random.rand() <= self.noise:
            # Apply noise: choose the action with no acceleration (actions[4])
            accel = self.actions[4]
        else:
            # If the agent is in the starting state, restrict to specific actions
            if any(np.array_equal(position, state) for state in self.starting_states):
                index = np.random.randint(1, 2)  # Restrict to specific action indices
                accel = self.actions[index]
            else:
                # Otherwise, use the provided action 'a'
                accel = self.actions[a]
        
        next_vel = self._vel_check(velocity,accel)
        next_pos = np.array([position[0]+next_vel[0],position[1]+next_vel[1]])
        # something is off here (velocity never really goes above 2 and 1 for x
        
        # if finish check, if accident_check, else next_pos = next_vel + pos or smth
        if self._accident_check(next_pos):
            return self._reset_pos(),reward,terminate
        elif self._finish_check(next_pos):
            terminate = True
            return State(next_pos,next_vel),0,terminate # reward might need to be 0? 
        else:
            return State(next_pos,next_vel),reward,terminate
        
        

def main():

    rt = RaceTrack(trackA=np.ones((32,17), dtype=float),trackB=np.ones((30,32), dtype=float))
    track_a = rt.generate_track_a()
    track_b = rt.generate_track_b()

    plt.figure(figsize=(10, 5))
    plt.imshow(track_b,cmap='Pastel1_r')
    plt.show()

    env= EnviornmentA(track=track_b)
    print(env.starting_states)
    print(env.ending_states)


if __name__=="__main__":
    main()
     




