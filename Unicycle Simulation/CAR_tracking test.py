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

# def compute_lyapunov_modelling_error(x,xstar,w_func,w_lb,w_ub,gamma):
#     assert x.shape == xstar.shape, "Trajectories x and xstar must have same shape"
#     length=x.shape[0]
#     # print(length)
#     error=np.zeros(length)
#     dist=np.zeros(length)

#     for i in range(length-1):
#         M_next = np.linalg.inv(w_func(x[i+1]))
#         M_curr = np.linalg.inv(w_func(x[i]))
#         dist[i]=np.linalg.norm(x[i]-xstar[i])**2
#         err= np.sqrt(w_lb/w_ub)*np.sqrt((x[i+1]-xstar[i+1]).T @ M_next @ (x[i+1]-xstar[i+1])) - gamma*(np.sqrt((x[i]-xstar[i]).T @ M_curr @ (x[i]-xstar[i])))  #(np.sqrt(m_ub/m_lb) *
#         # if err>=0:
#         #     error[i]=err
#         error[i]=err
#     return error,dist


# SMALL_SIZE = 8
# MEDIUM_SIZE = 10
# BIGGER_SIZE = 13
# HUGE_SIZE = 25

# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=HUGE_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=HUGE_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=15)    # fontsize of the tick labels
# plt.rc('legend', fontsize=20)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('axes', axisbelow=True)

left = 0.14  # the left side of the subplots of the figure
right = 0.98   # the right side of the subplots of the figure
bottom = 0.17  # the bottom of the subplots of the figure
top = 0.925     # the top of the subplots of the figure


system = importlib.import_module('system_CAR')
f, B, _, num_dim_x, num_dim_control = get_system_wrapper(system)
controller = get_controller_wrapper('../log_CAR_0.5_10_1/controller_best.pth.tar')      # 1<=CAR2<=2   0.1<=CAR3<=0.2 
w_func = get_w_func_wrapper_from_checkpoint("../log_CAR_0.5_10_1/model_best.pth.tar")
w_lb=0.5
w_ub=10




if __name__ == '__main__':
    config = importlib.import_module('config_CAR')
    time_bound = 10.
    time_step = 0.05
    t = np.arange(0, time_bound, time_step)
    num_realizations = 20

    #Get reference trajectories for PYTHON path planning
    data =  np.load('CAR_ref_U0.1.npz')
    # print(data.files)
    xstar = data['X'].T
    ustar = data['U'].T
    xstar = [x.reshape(-1,1) for x in xstar]    
    ustar = [u.reshape(-1,1) for u in ustar]


    #Get reference trajectories for MATLAB path planning
    # data = scipy.io.loadmat('x_star_GMM0.1.mat')
    # xstar=np.array(data['x_star']).T
    # xstar = [x.reshape(-1,1) for x in xstar]
    # data = scipy.io.loadmat('u_star_GMM0.1.mat')
    # ustar=np.array(data['u_star']).T    
    # ustar = [u.reshape(-1,1) for u in ustar]
    # print(ustar)
    # print(np.array(xstar).shape)
    # print(np.array(ustar).shape)
    # print(w_func(np.array([0,0.4,0,0])))
    # M_matrix =  np.linalg.inv(w_func(np.array([0,0.4,0,0])))
    # eigvals = np.linalg.eigvalsh(M_matrix)  # since M is symmetric
    # lambda_max,lambda_min = np.max(eigvals), np.min(eigvals)
    # print("m_lb :",lambda_min,"m_ub :",lambda_max)

    x_closed = []
    controls = []
    errors = []
    xinits = []


    # XE_INIT_MIN = np.array([-0.3,]*4)
    # XE_INIT_MAX = np.array([ 0.3,]*4)
    # xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
    xinit = xstar[0] #+ xe_0.reshape(-1,1)
    # a_vec=np.array([0,1,0,0])

    # print(w_func(xinit))
    fig = plt.figure(figsize=(5.0, 8.0))
    xstar=np.array(xstar)
    plt.plot(xstar[:,1,0],xstar[:,0,0],'k',label='reference')

    gamma = np.sqrt(1-(2*time_step*1)) * np.sqrt(w_lb/w_ub)
    print("gamma :",gamma)
    s= 0.3
    #Gaussian S=0.1 0.32   
    # Gaussian S0.05 0.211 
    #Unifrom S=0.1 0.301
    #Unifrom S=0.1 for gamma = 0.21  0.167
    #GMM 0.5747 0.822

    x_closed_all = []
    controls_all = []



    for i in range(num_realizations):
        x, u, omega_CP_sumterm = EulerIntegrate(controller, f, B, xstar,ustar,xinit,w_func,time_bound,time_step,gamma,with_tracking=True,sigma=0.1)
        x=np.array(x)
        u=np.array(u)
        x_closed_all.append(x)
        controls_all.append(u)
        # error, dist = compute_lyapunov_modelling_error(x[:,:,0],xstar[:,:,0],w_func,w_lb,w_ub,gamma)
        # print("Lyapunov modelling error :",np.max(error))
        plt.plot(x[:-1,1,0],x[:-1,0,0],'b', linewidth=1) #,label='closed-loop traj')   #0.8
        plt.plot(x[-1,1,0],x[-1,0,0],'r')
        
    # delta_v=  np.max(error)


    # Save all trajectories to a single file
    # np.savez('CAR_U0.1_smaller.npz',
    #          x_trajs=np.array(x_closed_all, dtype=object),
    #          u_trajs=np.array(controls_all, dtype=object),
    #          xstar=xstar,
    #          ustar=ustar,
    #          t=t)

    # PLOTTING
    orange = [0.8500, 0.3250, 0.0980]
    yellow = [0.9290, 0.6940, 0.1250]
    purple = [0.4940, 0.1840, 0.5560]
    olive_green = [0.4660, 0.8740, 0.1880]
    maroon = [0.6350, 0.0780, 0.1840]
    gray = [0.7, 0.7, 0.7]
    bad_blue = [0, 0.4470, 0.7410]



    # initial_dist = np.sqrt(((x_closed[0] - xstar[0])**2).sum())
    # errors.append([np.sqrt(((x-xs)**2).sum()) / initial_dist for x, xs in zip(x_closed[:-1], xstar)])


    # Start and goal markers
    plt.plot(0.4,0,  'go', markersize=5, markerfacecolor='g', label='Start')
    plt.plot(0.4,10,  'ro', markersize=5, markerfacecolor='r', label='Goal')

    # Plot circular obstacle
    obs_center = np.array([5, 0])
    obs_radius = 1.2
    theta = np.linspace(0, 2*np.pi, 100)
    xunit = obs_radius * np.cos(theta) + obs_center[0]
    yunit = obs_radius * np.sin(theta) + obs_center[1]
    plt.fill( yunit,xunit, color=gray, edgecolor='k')

    # Plot wall (rectangle from x=0 to 10, y=2 to 5)
    plt.fill([2, 2, 5, 5],[0, 10, 10, 0],  color=gray, edgecolor='k')


    plt.xlabel('y')
    plt.ylabel('x')
    plt.axis('equal')
    # plt.grid(True)
    # plt.xlim([-4, 5])
    # plt.ylim([-2, 5])
    # plt.legend()
    # plt.title('x-y Trajectory with Obstacle and Wall')

    # Set view to match MATLAB's view([90 -90]) (rotates plot)
    # plt.gca().invert_yaxis()  # flips y-axis downward

    theta = np.linspace(0, 2*np.pi, 100)
    circle = np.vstack((np.cos(theta), np.sin(theta)))  # shape (2, 100)

    # s= 0.216   #S0.05 0.9735  #S0.01 0.177
    for i in range(0,xstar[:,1,0].shape[0],2):
        # fac= np.sqrt(1/w_lb)* q + delta_v*((1-gamma**i)/(1-gamma))
        mu = xstar[i,0:2,0]
        # Sigma = np.array(w_func(xstar[i,:,0]))[:2,:2]
        Sigma = np.array(w_func(xstar[0,:,0]))[:2,:2]
        # Sigma = np.eye(2)
        # dist = np.sqrt((xstar[i,0,0]-5)**2 + (xstar[i,1,0]-0)**2 )
        # n_vec = (1 - 1.2/dist)*(1/(dist-1.2))* np.array([np.array(xstar[i,0,0])-5,np.array(xstar[i,1,0])-0,0,0]);


        # Eigen decomposition
        D, V = np.linalg.eigh(s**2 *Sigma)
        ellipse = V @ np.diag(np.sqrt(D)) @ circle
        ellipse += mu[:, None]

        # Plot ellipsoid
        plt.plot(ellipse[1, :], ellipse[0, :], 'r', linewidth=0.5)

    plt.title("w ~ GMM")
    plt.show()


    # PLOT CONTROLS WITH BOUNDS
    # plt.figure()
    # for u in controls_all:
    #     plt.plot(t, u[:,0], 'b')
    #     plt.plot(t, u[:,1], 'r')

    # # print(np.array(ustar).shape)

    # #plot reference control trajectory
    # plt.plot(t, np.array(ustar)[:-1,0,0], 'k')
    # plt.plot(t, np.array(ustar)[:-1,1,0], 'k')

    # for a in [-1, 1, -10, 10]:
    #     plt.axhline(y=a, color='g', linestyle=':')
    # plt.show()



