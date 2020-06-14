import numpy as np
import random
from tqdm import tqdm #This provides progress bar

results = np.array([0, 0, 0, 0, 0]) 
averages = np.array([0])
deviations = np.array([0]) #Empty array to be filled with data

for a in tqdm(range(20, 61)): #The number of cars, N to simulate
    countlist = [] #Empty list to be filled 
    cars = a
    for o in range(0, 5): #5 values of flow
        RL = 100
        L = 2*RL
        
        y = np.zeros(cars)
        pre_x = []
        
        while len(pre_x) < cars: #Random positions of cars
            r = random.randint(0 - RL, RL)
            if r not in pre_x:
                pre_x.append(r)
                
        pre_x2 = np.array(pre_x)
        x = np.sort(pre_x2) #Code doesn't work unless positions are sorted
        speed = np.zeros(cars) #cars initially start from rest
        
        vmax = 5 #maximum speed of the cars
        counter = 0
        
        p = 0.6 #probability for cars to slow down due to events such as someone crossing the road etc

	#In this project, both p and vmax were varied to obtain graphs of traffic flow against number density of cars
        
        for n in range(400): #timesteps
            for i in range(0, cars):
                        
                for m in range(0, cars): #accelerate rule
                    if speed[m] < vmax:
                        speed[m] = speed[m] + 1
                
                
                
                for k in range(0, cars - 1):
                    if x[k] > x[k + 1]:
                        d = L - (x[k] - x[k + 1])
                    else:
                        d = x[k + 1] - x[k]  #define distance between adjacent cars
                            
                    if speed[k] >= d:
                        speed[k] = d - 1 #prevents cars crashing into eachother
                    
                if x[cars - 1] > x[0]:
                    dL = L - (x[cars - 1] - x[0])
                else:
                    dL = x[0] - x[cars - 1]
        
                if speed[cars - 1] >= dL:
                    speed[cars - 1] = dL - 1
                    
                if speed[i] > 0:
                    j = random.randint(0, 100)
                    if j <= p*100:
                        speed[i] = speed[i] - 1
                    
                
                if x[i] + speed[i] > RL:
                    x[i] = x[i] + speed[i] - L
                else:
                    x[i] = x[i] + speed[i]
                
                
                
                if (x[i] - speed[i]) <= -75 < x[i]:
                    counter = counter + 1
        
                if (x[i] - speed[i]) <= -50 < x[i]:
                    counter = counter + 1
                    
                if (x[i] - speed[i]) <= -25 < x[i]:
                    counter = counter + 1            
                    
                if (x[i] - speed[i]) <= 0 < x[i]:
                    counter = counter + 1
                
                if (x[i] - speed[i]) <= 25 < x[i]:
                    counter = counter + 1
                    
                if (x[i] - speed[i]) <= 50 < x[i]:
                    counter = counter + 1
                    
                if (x[i] - speed[i]) <= 75 < x[i]:
                    counter = counter + 1            
        
        counter = counter/2800 #Counter is the average number of cars that passes a certain point along the road - determines traffic flow
        print(counter)
        countlist.append(counter)
    
    print(cars, countlist)
    results = np.vstack([results, countlist])
    mean = np.mean(countlist)
    dev = np.std(countlist)
    averages = np.vstack([averages, mean])
    deviations = np.vstack([deviations, dev])
    
print(results)
print(averages)

results2 = np.hstack([results, averages])
results3 = np.hstack([results2, deviations])

print(results3)
np.savetxt('results4', results3)




'''the below plots graphs of traffic flow against car number density (not actuaally part of the same program)'''

import numpy as np
import matplotlib.pyplot as plt

column = np.loadtxt('p0.4v8 peak')

y = column[1:, 5]
dev = column[1:, 6]

N = np.arange(20, 61)
x = N/200

plt.errorbar(x = x, y = y, yerr = dev, marker = 'x', linestyle = '-', color = 'blue')
plt.xlabel('Number density (Cars/Unit length)')
plt.ylabel('Traffic flow (cars/s)')
plt.suptitle('Traffic flow against density peak investigation for p = 0.4, vmax = 8')


plt.show()

'''the below plots multiple curves on one graph (again, not originally part of same program'''
import numpy as np
import matplotlib.pyplot as plt

column1 = np.loadtxt('p0.4v3 peak')
column2 = np.loadtxt('p0.4v4 peak')
column3 = np.loadtxt('p0.4v5 peak')
column4 = np.loadtxt('p0.4v6 peak')
column5 = np.loadtxt('p0.4v7 peak')
column6 = np.loadtxt('p0.4v8 peak')

y1 = column1[1:, 5]
dev1 = column1[1:, 6]

y2 = column2[1:, 5]
dev2 = column2[1:, 6]

y3 = column3[1:, 5]
dev3 = column3[1:, 6]

y4 = column4[1:, 5]
dev4 = column4[1:, 6]

y5 = column5[1:, 5]
dev5 = column5[1:, 6]

y6 = column6[1:, 5]
dev6 = column6[1:, 6]

N = np.arange(20, 61)
x = N/200

plt.plot(x, y1, label = 'vmax = 3')
plt.plot(x, y2, label = 'vmax = 4')
plt.plot(x, y3, label = 'vmax = 5')
plt.plot(x, y4, label = 'vmax = 6')
plt.plot(x, y5, label = 'vmax = 7')
plt.plot(x, y6, label = 'vmax = 8')

plt.legend(loc = 'upper right')

plt.xlabel('Number density (cars/unit length)')
plt.ylabel('Traffic flow (cars/timestep)')
plt.suptitle('Traffic flow against density peak investigation for p = 0.4, varying vmax')


plt.show()

'''below identifies peaks'''
import numpy as np

column1 = np.loadtxt('p0.4v3 peak')
column2 = np.loadtxt('p0.4v4 peak')
column3 = np.loadtxt('p0.4v5 peak')
column4 = np.loadtxt('p0.4v6 peak')
column5 = np.loadtxt('p0.4v7 peak')
column6 = np.loadtxt('p0.4v8 peak')

y1 = column1[1:, 5]
y2 = column2[1:, 5]
y3 = column3[1:, 5]
y4 = column4[1:, 5]
y5 = column5[1:, 5]
y6 = column6[1:, 5]

max1 = np.amax(y1)
max2 = np.amax(y2)
max3 = np.amax(y3)
max4 = np.amax(y4)
max5 = np.amax(y5)
max6 = np.amax(y6)

x1 = np.where(y1 == max1)
x2 = np.where(y2 == max2)
x3 = np.where(y3 == max3)
x4 = np.where(y4 == max4)
x5 = np.where(y5 == max5)
x6 = np.where(y6 == max6)

print(x1, max1)
print(x2, max2)
print(x3, max3)
print(x4, max4)
print(x5, max5)
print(x6, max6)