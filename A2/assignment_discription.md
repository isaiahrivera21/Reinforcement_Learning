Consider driving a race car around a turn like those shown in Figure 5.5. 
ou want to go as fast as possible, but not so fast as to run offthe track.
In our simplified racetrack, the car is at one of a discrete set of grid positions, the cells in the diagram.
The velocity is also discrete, a number of grid cells moved horizontally and vertically per time step.
Actions are increments to the velocity components
Each may be changed by +1, 1, or 0 in each step, for a total of
nine (3x3) actions. 
(Each what I think horizontal and vertically) (At an action we can choose whether to increment the velocity or not)
Rewards: -1 until thye car crosses the finish line

Have to get the projected path of the car (what is that like the next state?)We are going to have a 10% chance that the velocity does not increment at all. 

Use a Monte Carlo Control Method to compute the optimal policy FROM EACH STARTING STATE. 


