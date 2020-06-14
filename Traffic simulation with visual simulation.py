import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

RL = 10
L = 2*RL

cars = 5
y = np.zeros(cars)
pre_x = []

while len(pre_x) < cars:
    r = random.randint(0 - RL, RL)
    if r not in pre_x:
        pre_x.append(r)
        
pre_x2 = np.array(pre_x)
x = np.sort(pre_x2)
speed = np.zeros(cars)

vmax = 5
counter = 0

for i in tqdm(range(100)):
    for i in range(0, cars):
        plt.clf()
        plt.axis([0 - RL, RL, -6, 6])
        plt.grid()
        plt.axvline(x = 0)
        plt.axvline(x = 5)
        plt.axvline(x = -5)
        
        for j in range(0, cars):
            if j > cars/2:
                col = 'blue'
            else:
                col = 'red'
            plt.plot(x[j], y[j], col, marker = 'o', linestyle = ' ')
            
        plt.suptitle('Counter = ' + str(counter))
        plt.pause(0.000000000001)
        
        
        
        for m in range(0, cars):
            if speed[m] < vmax:
                speed[m] = speed[m] + 1
        
        
        
        for k in range(0, cars - 1):
            if x[k] > x[k + 1]:
                d = L - (x[k] - x[k + 1])
            else:
                d = x[k + 1] - x[k] 
                    
            if speed[k] >= d:
                speed[k] = d - 1
            
        if x[cars - 1] > x[0]:
            dL = L - (x[cars - 1] - x[0])
        else:
            dL = x[0] - x[cars - 1]

        if speed[cars - 1] >= dL:
            speed[cars - 1] = dL - 1
            
        
        
        if x[i] + speed[i] > RL:
            x[i] = x[i] + speed[i] - L
        else:
            x[i] = x[i] + speed[i]
            
        if (x[i] - speed[i]) <= -5 < x[i]:
            counter = counter + 1
            
        if (x[i] - speed[i]) <= 0 < x[i]:
            counter = counter + 1
        
        if (x[i] - speed[i]) <= 5 < x[i]:
            counter = counter + 1