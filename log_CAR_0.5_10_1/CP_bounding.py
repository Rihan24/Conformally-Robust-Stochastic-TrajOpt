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
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')
import argparse

def compute_lyapunov_modelling_error(x,xstar,w_func,w_lb,w_ub,gamma):
    assert x.shape == xstar.shape, "Trajectories x and xstar must have same shape"
    length=x.shape[0]
    # print(length)
    error=np.zeros(length)
    dist=np.zeros(length)

    for i in range(length-1):
        M_next = np.linalg.inv(w_func(x[i+1]))
        M_curr = np.linalg.inv(w_func(x[i]))
        dist[i]=np.linalg.norm(x[i]-xstar[i])**2
        err= np.sqrt(w_ub/w_lb)*np.sqrt((x[i+1]-xstar[i+1]).T @ M_next @ (x[i+1]-xstar[i+1])) - gamma*(np.sqrt((x[i]-xstar[i]).T @ M_curr @ (x[i]-xstar[i])))  #(np.sqrt(m_ub/m_lb) *
        # if err>=0:
        #     error[i]=err
        error[i]=np.max([0,err])
        
    return error,dist
def set_axes_equal(ax):
    """Make 3D plot axes have equal scale."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


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


system = importlib.import_module('system_QUADROTOR_9D')
f, B, _, num_dim_x, num_dim_control = get_system_wrapper(system)
controller = get_controller_wrapper('log_QUADROTORM_R100_0.5_25_0.8/controller_best.pth.tar')      # 1<=CAR2<=2   0.1<=CAR3<=0.2 
w_func = get_w_func_wrapper_from_checkpoint(checkpoint_path='log_QUADROTORM_R100_0.5_25_0.8/model_best.pth.tar', w_lb=0.5, task = "QUADROTOR_9D")
w_lb=0.5
w_ub=25

time_bound = 5
time_step = 0.01
t = np.arange(0, time_bound, time_step)
num_realizations = 1
gamma = np.sqrt(1-(2*time_step*1)) * np.sqrt(w_lb/w_ub)
print("gamma :",gamma)
q=  0.05   # For gamma=0.67 S0.01sq 0.05   #So.o5sq 0.2327  #UW0.1 0.2593 #GMM0.1 0.21




if __name__ == '__main__':
    config = importlib.import_module('config_QUADROTOR_9D')


    #Get reference trajectories for MATLAB path planning
    data = np.load('quadN_traj1_T5.npz')
    xstar = data['X'].T
    ustar = data['U'].T
    # data = scipy.io.loadmat('x_star_GMM0.1.mat')
    # xstar=np.array(data['x_star']).T
    xstar = [x.reshape(-1,1) for x in xstar]
    # data = scipy.io.loadmat('u_star_GMM0.1.mat')
    # ustar=np.array(data['u_star']).T
    
    ustar = [u.reshape(-1,1) for u in ustar]

    #  
    # M_matrix =  np.linalg.inv(w_func(np.array([0,0.4,0,0])))
    # eigvals = np.linalg.eigvalsh(M_matrix)  # since M is symmetric
    # lambda_max,lambda_min = np.max(eigvals), np.min(eigvals)
    # print("m_lb :",lambda_min,"m_ub :",lambda_max)

    x_closed = []
    controls = []
    errors = []
    xinits = []


    XE_INIT_MIN = np.array([-0.1,]*9)
    XE_INIT_MAX = np.array([ 0.1,]*9)
    xe_0 = XE_INIT_MIN + np.random.rand(len(XE_INIT_MIN)) * (XE_INIT_MAX - XE_INIT_MIN)
    xinit = xstar[0] + xe_0.reshape(-1,1)
    # print(xinit)
    # a_vec=np.array([0,1,0,0])

    # print(w_func(xinit))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xstar=np.array(xstar)
    ax.plot(xstar[:,0,0],xstar[:,1,0],xstar[:,2,0],'k',label='reference')



    for i in range(num_realizations):
        x, u = EulerIntegrate(controller, f, B, xstar,ustar,xinit,time_bound,time_step,with_tracking=True,sigma=0.00)
        x=np.array(x)
        u=np.array(u)
        # print(u)
        error, dist = compute_lyapunov_modelling_error(x[:,:,0],xstar[:,:,0],w_func,w_lb,w_ub,gamma)
        print("Lyapunov modelling error :",np.max(error))
        ax.plot(x[:-1,0,0],x[:-1,1,0],x[:-1,2,0],'b--') #,label='closed-loop traj')
        ax.scatter(x[-1,0,0],x[-1,1,0],x[-1,2,0])
        
    # delta_v=  np.max(error)
    set_axes_equal(ax)
    plt.show()





    # PLOTTING
    orange = [0.8500, 0.3250, 0.0980]
    yellow = [0.9290, 0.6940, 0.1250]
    purple = [0.4940, 0.1840, 0.5560]
    olive_green = [0.4660, 0.8740, 0.1880]
    maroon = [0.6350, 0.0780, 0.1840]
    gray = [0.7, 0.7, 0.7]
    bad_blue = [0, 0.4470, 0.7410]




    # for i in range(0,xstar[:,1,0].shape[0],2):
    #     fac= np.sqrt(1/w_lb)* q + delta_v*((1-gamma**i)/(1-gamma))
    #     mu = xstar[i,0:2,0]
    #     Sigma = np.array(w_func(xstar[i,:,0]))[:2,:2]
    #     # Sigma = np.eye(2)
    #     # dist = np.sqrt((xstar[i,0,0]-5)**2 + (xstar[i,1,0]-0)**2 )
    #     # n_vec = (1 - 1.2/dist)*(1/(dist-1.2))* np.array([np.array(xstar[i,0,0])-5,np.array(xstar[i,1,0])-0,0,0]);


    #     # Eigen decomposition
    #     D, V = np.linalg.eigh(fac**2 *Sigma)
    #     ellipse = V @ np.diag(np.sqrt(D)) @ circle
    #     ellipse += mu[:, None]

    #     # Plot ellipsoid
    #     plt.plot(ellipse[1, :], ellipse[0, :], 'r', linewidth=0.5)

    # plt.title("w ~ GMM")
    # set_axes_equal(ax)
    # plt.show()


    # Example placeholders — replace with your data


    # fig, (ax1,ax2) = plt.subplots(2)
    # ax1.plot(t,x[:-1,2], 'b-', label='z')
    # ax1.legend()
    # ax2.plot(t,u[:,0], 'b-', label='f_dot')
    # ax2.plot(t,u[:,1], 'k-', label='phi_dot')
    # ax2.plot(t,u[:,2], 'r-', label='theta_dot')
    # ax2.legend()
    # plt.show()



# PLOT CONFIDENCE SETS

# theta = np.linspace(0, 2*np.pi, 100)
# circle = np.vstack((np.cos(theta), np.sin(theta)))  # shape (2, 100)

# # Plot
# plt.figure(figsize=(8, 6))
# plt.plot(xstar[:,1,0], xstar[:,0,0], 'b-', label='Trajectory')

# for i in range(0,xstar[:,1,0].shape[0],5):
#     mu = xstar[i,0:2,0]
#     Sigma = np.array(w_func(xstar[i,:,0]))[:2,:2]

#     # Eigen decomposition
#     D, V = np.linalg.eigh(Sigma)
#     ellipse = V @ np.diag(np.sqrt(D )) @ circle
#     ellipse += mu[:, None]

#     # Plot ellipsoid
#     plt.plot(ellipse[1, :], ellipse[0, :], 'r', linewidth=1)


# plt.axis('equal')
# plt.grid(True)
# plt.xlabel('y')
# plt.ylabel('x')
# plt.legend()
# plt.show()