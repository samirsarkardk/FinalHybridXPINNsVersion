from Aconfig import Config
from Aconfig import DEVICE
import torch
import numpy as np
import matplotlib.pyplot as plt

config = Config()


x_min = config.x_min
x_max = config.x_max
t_min = config.t_min
t_max = config.x_max

x_min.to(DEVICE), x_max.to(DEVICE), t_min.to(DEVICE), t_max.to(DEVICE)

t_mid = (t_max + t_min)/2

t_mid.to(DEVICE)

def subdomain1():

    x1_min = x_min
    x1_max = x_max

    t1_min = t_min
    t1_max = t_mid 

    return x1_min, x1_max, t1_min, t1_max


def subdomain2():
    x2_min = x_min
    x2_max = x_max
    t2_min = t_mid 
    t2_max = t_max

    return x2_min, x2_max, t2_min, t2_max





def subdomain3():

    # Generate candidate points
    x = torch.linspace(x_min,x_max,steps=1000,device=DEVICE).reshape(-1,1)
    t = torch.full_like(x,t_mid,device=DEVICE)

    return x,t
















