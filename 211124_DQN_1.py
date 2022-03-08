#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 09:19:46 2021

@author: enricosoranzo
"""
import copy
import math
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
import random
import sys
import time
time_0 = time.perf_counter()
import torch
from collections import deque
from matplotlib import pylab as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
# Set seed
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
# -------------------------------------
# Calculate utilisation
# of single slice
# -------------------------------------
def Util_loc():
    coh_loc = coh_1
    tanphi_loc = tanphi_1
    xm = (xslip[-1]+xslip[-2])/2
    ym = (yslip[-1]+yslip[-2])/2
    width_loc = xslip[-1]-xslip[-2]
    length_loc = width_loc/math.cos(alpha[-1])
    if xm < 0: # Left of the toe
        height_loc = -ym
    if xm >= 0 and xm < b: # Below slope
        height_loc = -ym+xm*h/b
    if xm >= b: # Right of the top
        height_loc = -ym+h
    weight_loc = unit_weight*height_loc*(width_loc)
    N_loc = weight_loc*math.cos(alpha[-1])

    point = Point(xm,ym)
    if soil1.contains(point) or soil1.touches(point) == True:
        coh_loc = coh_1
        tanphi_loc = tanphi_1
    elif soil2.contains(point) or soil2.touches(point) == True:
        coh_loc = coh_2
        tanphi_loc = tanphi_2
    elif soil3.contains(point) or soil3.touches(point) == True:
        coh_loc = coh_3
        tanphi_loc = tanphi_3
    # If the slip surface is deep or long
    # the method fails 
    # the last value is thus taken
    else:
        coh_loc = coh_loc
        tanphi_loc = tanphi_loc
    num_loc = coh_loc*length_loc*math.cos(alpha[-1])+N_loc*tanphi_loc*math.cos(alpha[-1])
    den_loc = N_loc*math.sin(alpha[-1])       
    if alpha[-1] != 0:       
        FoS_loc = num_loc/den_loc # Factor of safety
        util_loc = 1/FoS_loc
    else:
        util_loc = 0
    return util_loc
# -------------------------------------
# Calculate utilisation 
# from force equilibrium 
# by neglecting interslice forces
# -------------------------------------
def Util_Force_Eq():
    global N
    global coh,tanphi
    global length
    global weight
    global width
    numerator = [0] # Numerator in the utilisation equation
    denominator = [0] # Denominator in the utilisation equation
    height = [0] # Slice height
    weight = [0] # Slice weight
    width = [0]
    length = [0]
    coh = [0]
    tanphi = [0]
    N = [0] # Force normal to slice base
    for i in range(1,len(xslip)):
        xm = (xslip[i]+xslip[i-1])/2
        ym = (yslip[i]+yslip[i-1])/2
        width.append(xslip[i]-xslip[i-1])
        length.append(width[i]/math.cos(alpha[i]))
        if xm < 0: # Left of the toe
            height.append(-ym)
        if xm >= 0 and xm < b: # Below slope
            height.append(-ym+xm*h/b)
        if xm >= b: # Right of the top
            height.append(-ym+h)
        weight.append(unit_weight*height[i]*(width[i]))
        N.append(weight[i]*math.cos(alpha[i]))

        point = Point(xm,ym)
        if soil1.contains(point) or soil1.touches(point) == True:
            coh.append(coh_1)
            tanphi.append(tanphi_1)
        elif soil2.contains(point) or soil2.touches(point) == True:
            coh.append(coh_2)
            tanphi.append(tanphi_2)
        elif soil3.contains(point) or soil3.touches(point) == True:
            coh.append(coh_3)
            tanphi.append(tanphi_3)
        # If the slip surface is deep or long
        # the method fails 
        # the last value is thus taken
        else:
            coh.append(coh[-1])
            tanphi.append(tanphi[-1])
          
        numerator.append(coh[i]*length[i]*math.cos(alpha[i])+N[i]*tanphi[i]*math.cos(alpha[i]))
        denominator.append(N[i]*math.sin(alpha[i]))       
    FoS = np.array(numerator).sum()/np.array(denominator).sum() # Factor of safety
    util = 1/FoS
    return util
# -------------------------------------
# Calculate utilisation 
# according to Janbu simplified
# from force equilibrium 
# by neglecting interslice shear forces
# but retaining interslice normal forces
# -------------------------------------
def Janbu(util):
    numerator = [0] # Numerator in the utilisation equation
    denominator = [0] # Denominator in the utilisation equation
    m_alpha = [0]
    N = [0] # Force normal to slice base
    for i in range(1,len(xslip)):
        m_alpha.append(math.cos(alpha[i])+util*math.sin(alpha[i])*tanphi[i])
        N.append((weight[i]-coh[i]*length[i]*math.sin(alpha[i])*util)/m_alpha[i])
        numerator.append(coh[i]*length[i]*math.cos(alpha[i])+N[i]*tanphi[i]*math.cos(alpha[i]))
        denominator.append(N[i]*math.sin(alpha[i]))       
    FoS = np.array(numerator).sum()/np.array(denominator).sum() # Factor of safety
    util = 1/FoS    
    return util
# -------------------------------------
# Calculate utilisation
# first by force equlibrium without interslice forces
# then with iteration of the Janbu simplified method
# -------------------------------------
def iterate_util():
    mu.append(Util_Force_Eq())
    tol = 1e-3
    itermax = 1000
    for j in range(1,itermax+1):
        mu.append(Janbu(mu[-1])) 
        if abs(mu[-1] - mu[-2]) < tol:
            break

# Slope geometry 
b = 30 # Slope base width
h = 20

# Soil layers
soil1 = Polygon(np.array([[-10,0],[0,0],[30,20],[60,20],[60,-10],[-10,-10]]))
soil2 = Polygon(np.array([[-10,0],[0,0],[30,20],[60,20],[60,-10],[-10,-10]]))
soil3 = Polygon(np.array([[-10,0],[0,0],[30,20],[60,20],[60,-10],[-10,-10]]))

# Soil parameters
unit_weight = 18.82
coh_1 = 41.65
coh_2 = 41.65
coh_3 = 41.65
phi_1 = 15
phi_2 = 15
phi_3 = 15
phi_1 = math.radians(phi_1)    
phi_2 = math.radians(phi_2)    
phi_3 = math.radians(phi_3)    
tanphi_1 = math.tan(phi_1)
tanphi_2 = math.tan(phi_2)
tanphi_3 = math.tan(phi_3)

# Grid
step = 2 # Grid step size
step_x0 = 0.5 # x0 step size
xmin = -10
xmax = 60
ymin = -10
ymax = h
alphamin = -(math.pi/4-phi_3/2)
alphamax = math.pi/4+phi_1/2
lx = xmax - xmin
ly = ymax - ymin
xsteps = round(lx/step + 1)
ysteps = round(ly/step + 1)
steps = xsteps + ysteps

# Environment mode
#mode = 'static'
mode = 'random'

# -------------------------------------
# DNN Definition
# -------------------------------------
l1 = 3 # Number of states
l2 = 50 # Neurons in 1st hidden layer
l3 = 50 # Neurons in 2nd hidden layer
l4 = 7 # Number of actions
# Build DNN
model = torch.nn.Sequential(
    torch.nn.Linear(l1,l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2,l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3,l4)
    )
# Creates a second model by making an identical copy 
# of the original copy of the original Q-network model
model2 = model2 = copy.deepcopy(model) 
model2.load_state_dict(model.state_dict()) # Copies the parameters of the original model

# Model parameters
loss_fn = torch.nn.MSELoss() # Define loss function
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
gamma = 0.9 # Discount factor
epsilon = 1.0 # Initial probability of random actions
if mode == 'random':
    episodes = 2000 # Number of epochs
else:
    episodes = 50

# Initialise variables/vectors
losses = [] # Loss vector
# Synchronises the frequency parameters
# Every 5 steps we will copy the parameters
# of the model into model2
sync_freq = round(episodes/200) 
mem_size = round(episodes/100) # Set the total size of the experience replay memory
batch_size = round(episodes/200) # Set the mini-batch size
replay = deque(maxlen=mem_size) # Creates the memory replay as a deque list
j = 0 # Counter to update the TARGET network

# Episode tracker
x0_his = []
rew_his = [] # Reward history vector
mu = [] # Utilisation vector
mu_his = [] # Utilisation history
# Slip surface history
xslip_his = [] 
yslip_his = []
deep = 0 # Counts the number of slip surfaces that touch the bottom boundary
long = 0 # Counts the number of slip surfaces that touch the right boundary

# -------------------------------------
# Main training loop
# -------------------------------------
for i in range(1,episodes+1):
    # Episode definition
    # Initial state
    if mode == 'random':
        x0 = round(random.randint((xmin)/step_x0, 0)*step_x0, 2) # Initial status
    else:
        x0 = -1.25
    y0 = 0
        
    # Append histories
    x0_his.append(x0)
    # Initialise variables
    status = 1 # Game is on
    done = False
    tot_reward = 0
    action_ = math.degrees(alphamin)
    
    # -------------------------------------
    # Define slip surface
    # -------------------------------------
    xslip = [x0]
    yslip = [y0]
    alpha = [0]
    # Find status corresponding to coordinates   
    state_ = np.zeros((1,l1), dtype=float, order='C')
    state_[0,0] = (x0-xmin)/(xmax-xmin) 
    state_[0,1] = (y0-ymin)/(ymax-ymin) 
    state_[0,2] = 0 
    state1 = torch.from_numpy(state_).float()
    
    while (status == 1):
        j += 1
        qval = model(state1)
        qval_ = qval.data.numpy()
        
        # Select angle increase with epsilon-greedy policy                
        if (random.random() < epsilon): # Select action based on epsilon-greedy strategy
            daction_ = np.random.randint(0,l4)
        else:
            daction_ = np.argmax(qval_)            

        # Slice base angle always increasing 
        # to ensure convex slip surface
        action_ = action_ + daction_
        if action_ > math.degrees(alphamax):
            action_ = math.degrees(alphamax)
           
        dx = step
        dy = step*math.tan(math.radians(action_))
    
        # Append coordinates to slip surface vectors
        xslip.append(xslip[-1] + dx)
        yslip.append(yslip[-1] + dy)
        alpha.append(math.atan((yslip[-1]-yslip[-2])/(xslip[-1]-xslip[-2])))
        
        # Reward
        # equal to slice utlisation for every step except the last
        # equal to 100*utilisation for last step
        reward = Util_loc()
        # Check if slip surface reaches bottom boundary
        if yslip[-1] < ymin:
            deep += 1
            reward = -100
            mu.append(0)
            done = True
        # Check if slip surface reaches right boundary
        if xslip[-1] >= xmax:
            long += 1
            reward = -100
            mu.append(0)
            done = True
        # Check if slip surface reaches the soil surface
        # CASE 1: Slip surface before slope toe
        if xslip[-1] <= 0 and yslip[-1] >= 0 and done == False:
            slope = (yslip[-1]-yslip[-2])/(xslip[-1]-xslip[-2])
            intercept = yslip[-2] - slope*xslip[-2]
            xslip[-1] = intercept/(-slope)
            yslip[-1] = 0
            iterate_util() # Get utilisation
            reward = 100*mu[-1] # Calculate reward
            done = True
        # CASE 2: Slip surface emerges in slope
        if xslip[-1] > 0 and xslip[-1] < b and yslip[-1] >= xslip[-1]*h/b and done == False: 
            slope = (yslip[-1]-yslip[-2])/(xslip[-1]-xslip[-2])
            intercept = yslip[-2] - slope*xslip[-2]
            xslip[-1] = intercept/(h/b-slope)
            yslip[-1] = xslip[-1]*h/b
            iterate_util() # Get utilisation
            reward = 100*mu[-1] # Calculate reward
            done = True
        # CASE 3: Slip surface emerges after slope top
        if xslip[-1] >= b and yslip[-1] >= h and done == False: 
            xslip[-1] = xslip[-2] + (h-yslip[-2])*(xslip[-1]-xslip[-2])/(yslip[-1]-yslip[-2])
            yslip[-1] = h
            iterate_util()
            reward = 100*mu[-1]
            done = True
        # It might be that the correction of CASE 3
        # resulted in a slip surface that cuts the slope
        # Therefore, we need the following correction
        if xslip[-1] > 0 and xslip[-1] < b and yslip[-1] == h:
            xslip[-1] = b 
            yslip[-1] = h
            iterate_util()
            reward = 100*mu[-1]
            done = True
        # It might be that the surfaces with x > xmax
        # show also y > h. Therefore the last y must be set to h
        if yslip[-1] > h:
            slope = (yslip[-1]-yslip[-2])/(xslip[-1]-xslip[-2])
            intercept = yslip[-2] - slope*xslip[-2] 
            xslip[-1] = (h-intercept)/slope
            yslip[-1] = h

        state2_ = np.zeros((1,l1), dtype=float, order='C')
        state2_[0,0] = (xslip[-1]-xmin)/(xmax-xmin) 
        state2_[0,1] = (yslip[-1]-ymin)/(ymax-ymin) 
        state2_[0,2] = (math.atan(dy/dx)-alphamin)/(alphamax-alphamin) 
        state2 = torch.from_numpy(state2_).float()

        exp = (state1,daction_,reward,state2,done) # Creates an experience of state, reward, action and the next state
        replay.append(exp) # Adds the experience to the experience replay list
        state1 = state2
        # Randomly samples a subset of the replay list
        # if the replay list is at least as long as the mini-batch size
        # begins the mini-batch training
        if len(replay) > batch_size: 
            minibatch = random.sample(replay,batch_size) # Randomly samples a subset of the replay list
            # Separates out the components of each experience into separate mini-bacth tensors
            state1_batch = torch.cat([s1 for (s1,a,r,s2,d) in minibatch]) 
            action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
            reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
            state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
            done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])
        
            # Recomputes Q values for the mini-batch of states to get gradients
            Q1 = model(state1_batch)
            with torch.no_grad():
            # Uses the TARGET network to get the max Q value for the next state
                Q2 = model2(state2_batch) 
        
            # Computes Q values for the mini-batch of the next states, 
            # but does not compute gradients
            Y = reward + (gamma*(1 - done_batch)*torch.max(Q2,dim=1)[0])
            X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
            loss = loss_fn(X,Y.detach())
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            
            # Copies the main model parameters to the target network
            if j % sync_freq == 0:
                model2.load_state_dict(model.state_dict())

        # If the game is over reset status and mov number
        if done == True:
            status = 0
            mu_his.append(mu[-1])
        tot_reward = tot_reward + reward
    if epsilon > 0.1:
        epsilon -= (1/episodes)
    rew_his.append(tot_reward)
    xslip_his.append(xslip)
    yslip_his.append(yslip)
    print(str(i)+'/'+str(episodes)+  
          ' x0: ' + str(xslip[0]) + ' \u03BC: '+ str(round(mu[-1],3)))
losses = np.array(losses) # Converts to numpy array for plots

# Print results
print()
print('Average utilisation: ' + str(round(np.array(mu_his).mean(),3)))
print('StDev: ' + str(round(np.array(mu_his).std(),3)))
print('Max utilisation: ' + str(round(max(mu_his),3)))
print()
print('Long (%): ' + str(round(long/episodes*100,2)))
print('Deep (%): ' + str(round(deep/episodes*100,2)))

# Save results
# Input
if mode != 'random':
    f = open('DQN_Stat_Res_1.csv','w')
else:
    f = open('DQN_Rand_Res_1.csv','w')
f.write('i,x0,mu,xslip,yslip\n')
for i in range(0,episodes):
    f.write(str(i+1)+','+
            str(x0_his[i])+','+
            str(mu_his[i])+','+
            str(xslip_his[i])+','+
            str(yslip_his[i])+'\n')

# Plot slip surfaces
plt.figure()
crit_slope = np.array(mu_his).argmax() # Number of critical slope
mu_min = np.array(mu_his).min()
mu_max = np.array(mu_his).max()
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.xlim(xmin,xmax)
plt.ylim(ymin,40)
plt.xticks(family='Arial',size=10)
plt.yticks(family='Arial',size=10)
plt.xlabel('x (m)',family='Arial',size=10)
plt.ylabel('y (m)',family='Arial',size=10)            
# Plot soil layers
#plt.plot([4,xmax],[4/3,4/3], color = 'gray', linewidth = 1)
#plt.plot([-5,xmax],[-3,-3], color = 'gray', linewidth = 1)
# Plot all slip surfaces:
for i in range(0,len(xslip_his),1):
    mu_min = mu_max/2  
    mu_s = (max(mu_his[i],mu_min)-mu_min)/(mu_max-mu_min) 
    col_r = min(2*mu_s,1)
    col_g = min(2*(1-mu_s),1)
#    col_a = max(1,i/episodes)
    # if mu_his[i] > 0:
    #     plt.plot(xslip_his[i], yslip_his[i], linewidth=0.5, color=(col_r,col_g,0,0.5))
    plt.plot(xslip_his[i], yslip_his[i], linewidth=0.5, color=(col_r,col_g,0,0.5))
# Plot critical slip surface
plt.plot(xslip_his[crit_slope], yslip_his[crit_slope], linewidth=0.5, color=(0,0,0,1))
plt.text(38,40-4,'Critical slip surface', color=(0,0,0),family='Arial') 
plt.text(38,40-7,'$\mu$ = ' + str(round(mu_his[crit_slope],3)), color=(0,0,0),family='Arial') 
plt.text(38,40-10,'$F_s$ = ' + str(round(1/mu_his[crit_slope],3)), color=(0,0,0),family='Arial') 
plt.text(38,40-13,'$x_0$ = ' + str(x0_his[crit_slope]) + ' m', color=(0,0,0),family='Arial') 
plt.text(38,40-16,'$n_s$ = ' + str(len(xslip_his[crit_slope])-1), color=(0,0,0),family='Arial') 
# Plot slices
for i in range(len(xslip_his[crit_slope])):
    xslice = xslip_his[crit_slope][i]
    ybase = yslip_his[crit_slope][i]
    if xslice <= 0:
        ytop = 0
    elif xslice <= b:
        ytop = xslice/b*h
    elif xslice > b:
        ytop = h
    plt.plot([xslice,xslice],[ybase,ytop],linewidth=0.5, color=(0,0,0,1))
# Plot slope
plt.plot([xmin,0,b,xmax], [0,0,h,h], color = 'black', linewidth=1)
# Color bar
for cb in np.arange(20.4,37.6,0.1):
    y_s = (cb-20)/(38-20)
#    y_s = y_s**3
    col_r = min(2*y_s,1)
    col_g = min(2*(1-y_s),1)
    plt.plot([-7.6,-6.4],[cb,cb], color = (col_r,col_g,0,0.5), linewidth = 4)
plt.plot([-8,-6,-6,-8,-8],[20,20,38,38,20], color = 'black', linewidth = 1)
for cb_thick in np.arange(20,40,2):
    plt.plot([-6,-5.8],[cb_thick,cb_thick], color = 'black', linewidth = 1)
    if cb_thick !=20:
        mu_y = (mu_max-mu_min)/(38-20)*(cb_thick-20)+mu_min
        plt.text(-5.5,cb_thick-0.3, str(round(mu_y,3)), color='black', fontsize=7,family='Arial')
    else:
        plt.text(-5.5,cb_thick-0.3, '< ' + str(round(mu_min,3)), color='black', fontsize=7,family='Arial')
# Save picture to file
if mode != 'random':
    plt.savefig('DQN_Stat_Slip_1.pdf')
else:
    plt.savefig('DQN_Rand_Slip_1.pdf')    
plt.show()

# Plot utilisation
plt.figure()
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.xlim(0,episodes)
plt.ylim(0,round(max(mu_his),1)+0.1)
plt.xticks(family='Arial',size=10)
plt.yticks(family='Arial',size=10)
plt.xlabel('Episodes (-)',family='Arial',size=10)
plt.ylabel('$\mu$ (-)',family='Arial',size=10)
plt.plot(mu_his, linewidth=0.5)
# Rolling average
mu_his_roll = pd.DataFrame(mu_his)
mu_his_roll = mu_his_roll.iloc[:].rolling(window=round(episodes/10)).mean()
mu_his_roll_std = mu_his_roll.iloc[:].rolling(window=round(episodes/10)).std()
plt.plot(mu_his_roll, linewidth=0.5,color='black')
font = font_manager.FontProperties(family='Arial',
                                   size=10)
plt.legend(['Utilisation','Moving average'],fancybox=False,edgecolor='0',prop=font,loc='lower right',facecolor='white', framealpha=1)
if mode != 'random':
    plt.savefig('DQN_Stat_Util_1.pdf')
else:
    plt.savefig('DQN_Rand_Util_1.pdf')    
plt.show()

# Plot utilisation with valid surfaces only
mu_val = []
for i in mu_his:
    if i != 0:
        mu_val.append(i)
plt.figure()
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.xlim(0,round(len(mu_val),-2))
plt.ylim(round(min(mu_val),1)-0.1,round(max(mu_val),1)+0.1)
plt.xticks(family='Arial',size=10)
plt.yticks(family='Arial',size=10)
plt.xlabel('Episodes (-)',family='Arial',size=10)
plt.ylabel('$\mu$ (-)',family='Arial',size=10)
plt.plot(mu_val, linewidth=0.5)
# Rolling average
mu_val_roll = pd.DataFrame(mu_val)
mu_val_roll = mu_val_roll.iloc[:].rolling(window=round(episodes/10)).mean()
mu_val_roll_std = mu_val_roll.iloc[:].rolling(window=round(episodes/10)).std()
plt.plot(mu_val_roll, linewidth=0.5,color='black')
font = font_manager.FontProperties(family='Arial',
                                   size=10)
plt.legend(['Utilisation','Moving average'],fancybox=False,edgecolor='0',prop=font,loc='lower right',facecolor='white', framealpha=1)
if mode != 'random':
    plt.savefig('DQN_Stat_Util_Val_1.pdf')
else:
    plt.savefig('DQN_Rand_Util_Val_1.pdf')    
plt.show()

# Plot reward
plt.figure()
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.xlim(0,episodes)
plt.ylim(0,round(max(rew_his),-1)+10)
plt.xlabel('Episodes (-)',family='Arial',size=10)
plt.ylabel('Total reward (-)',family='Arial',size=10)
plt.xticks(family='Arial',size=10)
plt.yticks(family='Arial',size=10)
plt.plot(rew_his,linewidth=0.5)
# Rolling average
rew_his_roll = pd.DataFrame(rew_his)
rew_his_roll = rew_his_roll.iloc[:].rolling(window=round(episodes/10)).mean()
rew_his_roll_std = rew_his_roll.iloc[:].rolling(window=round(episodes/10)).std()
plt.plot(rew_his_roll, linewidth=0.5,color='black')
font = font_manager.FontProperties(family='Arial',
                                   size=10)
plt.legend(['Total reward','Moving average'],fancybox=False,edgecolor='0',prop=font,loc='lower right',facecolor='white', framealpha=1)
if mode != 'random':
    plt.savefig('DQN_Stat_Rew_1.pdf')
else:
    plt.savefig('DQN_Rand_Rew_1.pdf')    
plt.show()

# Plot loss
plt.figure()
for pos in ['right', 'top', 'bottom', 'left']:
    plt.gca().spines[pos].set_linewidth(1)
plt.plot(losses, linewidth=0.5)
plt.xticks(family='Arial',size=10)
plt.yticks(family='Arial',size=10)
# Rolling average
losses_roll = pd.DataFrame(losses)
losses_roll = losses_roll.iloc[:].rolling(window=round(len(losses)/10)).mean()
plt.plot(losses_roll, linewidth=0.5,color='black')
plt.legend(['Loss','Moving average'],fancybox=False,edgecolor='0',prop=font,loc='upper right',facecolor='white', framealpha=1)
plt.xlim(0,len(losses))
if mode == 'random':
    plt.ylim(0,round(max(losses),-3)+1000)
plt.xticks(family='Arial',size=10)
plt.yticks(family='Arial',size=10)
plt.xlabel('Epochs (-)',family='Arial',size=10)
plt.ylabel('Loss (-)',family='Arial',size=10)
if mode != 'random':
    plt.savefig('DQN_Stat_Loss_1.pdf')
else:
    plt.savefig('DQN_Rand_Loss_1.pdf')    
plt.show()

# Plot number of slices
num_slice_his = []
for i in range(len(xslip_his)):
    num_slice_his.append(len(xslip_his[i]))
plt.figure()
plt.plot(num_slice_his, linewidth=0.5)
# Rolling average
num_slice_roll = pd.DataFrame(num_slice_his)
num_slice_roll = num_slice_roll.iloc[:].rolling(window=round(len(num_slice_his)/10)).mean()
plt.plot(num_slice_roll, linewidth=0.5,color='black')
plt.xlim(0,len(num_slice_his))
plt.ylim(0,round(max(num_slice_his),-1)+10)
plt.xticks(family='Arial',size=10)
plt.yticks(family='Arial',size=10)
plt.xlabel('Epochs (-)',family='Arial',size=10)
plt.ylabel('Number of slices (-)',family='Arial',size=10)
plt.legend(['Number of slices','Moving average'],fancybox=False,edgecolor='0',prop=font,loc='upper right',facecolor='white', framealpha=1)
if mode != 'random':
    plt.savefig('DQN_Stat_Slices_1.pdf')
else:
    plt.savefig('DQN_Rand_Slices_1.pdf')    
plt.show()
    
time_1 = time.perf_counter()
print()
print('Time elapsed: ' + str(round(time_1-time_0,2))+' sec')
