from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io
from scipy.stats import chi2
from np2pth import get_system_wrapper, get_controller_wrapper, get_w_func_wrapper_from_checkpoint

import importlib
from utils import EulerIntegrate
import time

import os
import sys
sys.path.append('../systems')
sys.path.append('../configs')
sys.path.append('../models')
import argparse

def compute_lyapunov_modelling_error(x,xstar,ustar,w_func,controller, w_lb,w_ub,gamma,dt):
    # print(x.shape,xstar.shape)
    assert x.shape == xstar.shape, "Trajectories x and xstar must have same shape"
    length=x.shape[0]
    # print(length)
    # error=np.zeros(length)
    # dist=np.zeros(length)
    Lyapunov_CP_array = []
    Lyapunov_CP_sum = 0

    for i in range(length-1):
        # M_next = np.linalg.inv(w_func(xstar[i+1]))
        # M_curr = np.linalg.inv(w_func(xstar[i]))
        M_next = np.linalg.inv(w_func(xstar[0]))
        M_curr = np.linalg.inv(w_func(xstar[0]))
        # dist[i]=np.linalg.norm(x[i]-xstar[i])**2
        u= controller(x[i], x[i]- xstar[i], ustar[i])
        xnext = dynamics_onestep(x[i],u,dt)
        err= np.sqrt(w_ub/w_lb)*np.sqrt((xnext-xstar[i+1]).T @ M_next @ (xnext-xstar[i+1])) - gamma*(np.sqrt((x[i]-xstar[i]).T @ M_curr @ (x[i]-xstar[i])))  #(np.sqrt(m_ub/m_lb) *
        # if err>=0:
        #     error[i]=err
        deltaV_k=np.max([0,err])
        Lyapunov_CP_sum+= deltaV_k*(gamma**i)
        # Lyapunov_CP_array.append(Lyapunov_CP_sum)
        
    return  Lyapunov_CP_sum #np.array(Lyapunov_CP_array), Lyapunov_CP_sum

def dynamics_onestep(x,u,dt):
    xcurr = x.reshape(-1,1)
    dx = f(xcurr) + B(xcurr).dot(u.reshape(-1,1))
    xnext = xcurr + dx * dt
    return xnext.flatten()


def propagate_states(x0, f, B, ustar, dt):
    x_list = [x0.reshape(-1)]
    for u in ustar:
        xcurr = x_list[-1].reshape(-1,1)
        dx = f(xcurr) + B(xcurr).dot(u.reshape(-1,1))
        xnext = xcurr + dx * dt
        x_list.append(xnext.flatten())
    return np.array(x_list)

def generate_random_ustar(t, uref_min, uref_max, num_freqs=30):
    freqs = list(range(1, num_freqs + 1))
    weights = np.random.randn(len(freqs), len(uref_min))
    weights /= np.linalg.norm(weights, axis=0, keepdims=True)
    
    uref = []
    for _t in t:
        u = np.zeros(len(uref_min))
        for freq, weight in zip(freqs, weights):
            u += weight * np.sin(freq * _t / t[-1] * 2 * np.pi)
        # Clip to min/max bounds
        u = np.clip(u, uref_min.flatten(), uref_max.flatten())
        uref.append(u)
    return np.array(uref)


# TUNING PARAMETERS = w_lb, w_ub, gamma, num_freqs, (sigma,D), num_of_realizations

system = importlib.import_module('system_CAR')
f, B, _, num_dim_x, num_dim_control = get_system_wrapper(system)
controller = get_controller_wrapper('../log_CAR_0.5_10_1/controller_best.pth.tar')      # 1<=CAR2<=2   0.1<=CAR3<=0.2 
w_func = get_w_func_wrapper_from_checkpoint(checkpoint_path='../log_CAR_0.5_10_1/model_best.pth.tar', w_lb=0.5, task = "CAR")
w_lb=0.5
w_ub=10
gamma_CT=1

time_bound = 10
time_step = 0.05
t = np.arange(0, time_bound, time_step)

num_realizations = 20
num_freqs = 10
gamma_DT = np.sqrt(1-(2*time_step*gamma_CT)) * np.sqrt(w_lb/w_ub)

gamma =  gamma_DT
print("gamma :",gamma)

# q=  0.05   # For gamma=0.67 S0.01sq 0.05   #So.o5sq 0.2327  #UW0.1 0.2593 #GMM0.1 0.21





if __name__ == '__main__':
    config = importlib.import_module('config_CAR')


    #Get reference trajectories for PYTHON path planning
    # data = np.load('car.npz')
    # xstar = data['X'].T
    # # print(np.shape(xstar))
    # ustar = data['U'].T

    #Get reference trajectories for MATLAB path planning
    # data = scipy.io.loadmat('x_star_GMM0.1.mat')
    # xstar=np.array(data['x_star']).T
    # data = scipy.io.loadmat('u_star_GMM0.1.mat')
    # ustar=np.array(data['u_star']).T

    # Sample xstar and ustar randomaly
    UREF_MIN = np.array([-1., -10.]).reshape(-1,1)
    UREF_MAX = np.array([ 1.,  10.]).reshape(-1,1)
    XE_INIT_MIN = np.array([-0.5,]*4)
    XE_INIT_MAX = np.array([ 0.5,]*4)
    X_INIT_MIN = np.array([-2., -1., -0.5, 0])
    X_INIT_MAX = np.array([ 2.,  1.,  0.5, 0])

    # print(w_func(xinit))
    fig = plt.figure(figsize=(4.0, 4.0))
    s_array = np.zeros(num_realizations)
    all_s = [] 

    for i in range(num_realizations):
        # Sample initial reference state xstar[0]
        # xstar_0 = X_INIT_MIN + np.random.rand(len(X_INIT_MIN)) * (X_INIT_MAX - X_INIT_MIN)
        # xstar_0 = xstar_0.reshape(-1,1)
        xstar_0 = np.array([0,0.4,0,0]).reshape(-1,1)
        # xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
        xinit =  xstar_0      # xstar_0  #xstar_0[0] + xe_0.reshape(-1,1)

        ustar = generate_random_ustar(t, UREF_MIN, UREF_MAX, num_freqs)
        xstar = propagate_states(xinit, f, B, ustar, time_step)

        xstar = [x.reshape(-1,1) for x in xstar]    
        ustar = [u.reshape(-1,1) for u in ustar]
        

        xstar=np.array(xstar)
        ustar = np.array(ustar)
        plt.plot(xstar[:,1,0],xstar[:,0,0],'k--')

        #SIMULATE        
        # NOTE THAT FOR SIMULATION XINIT IS SAME AS XREF XINIT
        # xinit = xstar_0
        x, u, omega_CP_sumterm = EulerIntegrate(controller, f, B, xstar,ustar,xinit,w_func,time_bound,time_step,gamma, with_tracking=True,sigma=0.1)
        x=np.array(x)   #[:-1,:,:]
        u=np.array(u)    #[:-1,:,:]
        # print(u)
        lyapunnov_CP_term= compute_lyapunov_modelling_error(x[:,:,0],xstar[:,:,0],ustar[:,:,0],w_func,controller, w_lb,w_ub,gamma,time_step)

        # print("Lyapunov modelling error :",lyapunnov_CP_term)
        # print('Uncertainty norm CP term :',omega_CP_sumterm )    #np.sqrt(1/w_lb)*omega_CP_sumterm )
        s  = lyapunnov_CP_term + omega_CP_sumterm
        # sum_array = lyapunov_CP_array + omega_CP_array
        print("Nonconformity score = ",s, "\n" )

        s_array[i] = s
        # all_s.append(sum_array)

        #PLOT CLOSED-LOOP Trajectories
        plt.plot(x[:,1,0],x[:,0,0],'b')
        plt.plot(x[0,1,0],x[0,0,0],  'go', markersize=5, markerfacecolor='g')
        plt.plot(x[-1,1,0],x[-1,0,0],  'ro', markersize=5, markerfacecolor='r')


    #CALCULATE QUANTILE FOR CP BOUND
    p = 0.9 # desired quantile level, e.g., 90%
    q = np.quantile(np.append(s_array, np.inf), p)
    print(f"Quantile at level {p}:", q)

    # quantiles = np.quantile(all_s, p, axis=0).flatten()






    # PLOTTING
    # orange = [0.8500, 0.3250, 0.0980]
    # yellow = [0.9290, 0.6940, 0.1250]
    # purple = [0.4940, 0.1840, 0.5560]
    # olive_green = [0.4660, 0.8740, 0.1880]
    # maroon = [0.6350, 0.0780, 0.1840]
    # gray = [0.7, 0.7, 0.7]
    # bad_blue = [0, 0.4470, 0.7410]

    # plt.plot(0.4,0,  'go', markersize=5, markerfacecolor='g', label='Start')
    # plt.plot(0.4,10,  'ro', markersize=5, markerfacecolor='r', label='Goal')

    # # Plot circular obstacle
    # obs_center = np.array([5, 0])
    # obs_radius = 1.2
    # theta = np.linspace(0, 2*np.pi, 100)
    # xunit = obs_radius * np.cos(theta) + obs_center[0]
    # yunit = obs_radius * np.sin(theta) + obs_center[1]
    # plt.fill( yunit,xunit, color=gray, edgecolor='k')

    # # Plot wall (rectangle from x=0 to 10, y=2 to 5)
    # plt.fill([2, 2, 5, 5],[0, 10, 10, 0],  color=gray, edgecolor='k')


    # plt.xlabel('y')
    # plt.ylabel('x')
    plt.xticks([])   # removes x-axis ticks
    plt.yticks([]) 
    plt.tight_layout()
    plt.axis('equal')

    # plt.legend()
    plt.show()
