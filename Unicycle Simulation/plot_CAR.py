import numpy as np
import scipy.io as sio
from scipy.stats import chi2
import matplotlib.pyplot as plt
from np2pth import get_system_wrapper, get_controller_wrapper, get_w_func_wrapper_from_checkpoint
import importlib
import sys
sys.path.append('../systems')
sys.path.append('../configs')
sys.path.append('../models')


def collision_percentage_GP(x_trajs: np.ndarray) -> float:

    # Extract x and y
    x = x_trajs[0, :, :]  # shape (N, K)
    y = x_trajs[1, :, :]  # shape (N, K)

    # Constraint violations
    violation1 = y > 2
    violation2 = np.sqrt((x - 5)**2 + y**2) < 1.2

    # Any violation along time for each trajectory
    violations = np.any(violation1 | violation2, axis=0)  # shape (K,)

    # Percentage of violating trajectories
    K = x_trajs.shape[2]
    return 100.0 * np.sum(violations) / K

def collision_percentage(x_trajs):
    num_violated = 0
    total = len(x_trajs)
    
    for traj in x_trajs:
        # Ensure each traj is a numeric array
        traj = np.asarray(traj, dtype=float)
        
        x = traj[:, 0, 0]
        y = traj[:, 1, 0]
        
        # Constraint 1: y <= 2
        violation1 = np.any(y > 2)
        
        # Constraint 2: distance from (5,0) >= 1.2
        violation2 = np.any(np.sqrt((x - 5)**2 + y**2) < 1.2)
        
        if violation1 or violation2:
            num_violated += 1
    
    return 100.0 * num_violated / total

# Load DATA
test = [
        # 'CAR_S0.05', 
        # 'CAR_S0.1',
        'CAR_U0.1',
        'CAR_GMM0.05'
]
label_name = [
    'Uniform',                 #r'$\sin(x)$'
    'GM'
]
s= [
    # 0.211, 
    # 0.32 ,
    0.3 ,   #0.176,
    0.5747   #0.645        #0.822
]
#Gaussian S=0.1 0.32   
# Gaussian S0.05 0.211 
# #Unifrom S=0.1 0.301
#GMM 0.822

orange = [0.8500, 0.3250, 0.0980]
yellow = [0.9290, 0.6940, 0.1250]
purple = [0.4940, 0.1840, 0.5560]
olive_green = [0.4660, 0.8740, 0.1880]
maroon = [0.6350, 0.0780, 0.1840]
gray = [0.7, 0.7, 0.7]
bad_blue = [0, 0.4470, 0.7410]
teal        = [0.3010, 0.7450, 0.9330]  # light blue / cyan
pink        = [0.8500, 0.3250, 0.6000]  # soft magenta-pink
lime        = [0.4660, 0.6740, 0.1880]  # darker green (less neon than olive)
gold        = [0.9290, 0.8940, 0.1250]  # lighter yellow variant
navy        = [0.0000, 0.2470, 0.5410]  # dark blue complement

colors= [ 
    navy,
    # olive_green
    maroon,
    # orange
]



# data = np.load('CAR_S.npz', allow_pickle=True)
# x_trajs = data['x_trajs']
# u_trajs = data['u_trajs']
# xstar = data['xstar']
# ustar = data['ustar']
# t = data['t']

# GP_planning matlab data'
gp_test = {
            'GP-LQR Matlab/GP2_for_GMM0.05',
            'GP-LQR Matlab/GP2_for_U0.1'



}
# data = sio.loadmat('GP_gaussian.mat')
# traj_true = data['traj_true']  # shape (4, N, P)
# X_nom = data['X']              # nominal states
# U_nom = data['U']              # nominal controls




###########################################

# system = importlib.import_module('system_CAR')
# f, B, _, num_dim_x, num_dim_control = get_system_wrapper(system)
# w_func = get_w_func_wrapper_from_checkpoint(checkpoint_path='log_CAR_0.5_10_1/model_best.pth.tar', w_lb=0.5, task = "CAR")
Sigma = np.array([
    [ 1.536628,   -0.00522158],
    [-0.00522158,  1.7324215 ]
])     
#np.array(w_func(np.array([0, 0.4, 0, 0])))[:2,:2]
# print(Sigma)



theta = np.linspace(0, 2*np.pi, 100)
circle = np.vstack((np.cos(theta), np.sin(theta)))


fig = plt.figure(figsize=(4.0, 6.0))
# for name, sval,c, labe in zip(test, s,colors, label_name):
#     data = np.load(f'{name}.npz', allow_pickle=True)
#     xstar = data['xstar']
#     x_trajs = data['x_trajs']

#     #Plot realizations
#     for x in x_trajs:
#         plt.plot(x[:-1, 1, 0], x[:-1, 0, 0],'-',color=c, linewidth=0.5)
#     plt.plot(x[-1, 1, 0], x[-1, 0, 0],'-',color=c, linewidth=0.5, label=f'{labe}')

#     #plot ellipsoids
#     for i in range(0,xstar[:,1,0].shape[0],4):
#         mu = xstar[i,0:2,0]
#         # Sigma = np.array(w_func(xstar[i,:,0]))[:2,:2]
#         D, V = np.linalg.eigh(sval**2 *Sigma)
#         ellipse = V @ np.diag(np.sqrt(D)) @ circle
#         ellipse += mu[:, None]
#         plt.plot(ellipse[1, :], ellipse[0, :], color=c, linewidth=0.4)
    
#     # plot reference trajectory with label
#     plt.plot(xstar[:-1, 1, :], xstar[:-1, 0, :],'--',color='k',linewidth=1.0)    #label=f'{name}, s={sval:.3f}'


#     x_trajs_array = np.stack([np.asarray(traj) for traj in x_trajs], axis=0)

#     coll = collision_percentage(x_trajs_array)
#     print(name, coll)

    


##################################################
#GP plots
p = 1-0.1/2
s = chi2.ppf(p, df=2)
colors= [ 
    # bad_blue,
    # olive_green
    navy,
    maroon
    # orange
]

for traj, c, labe in zip(gp_test,colors, label_name):
    data = sio.loadmat(f'{traj}.mat')
    traj_true = data['traj_true']  # shape (4, N, P)
    X_nom = data['X']              # nominal states
    # U_nom = data['U']              # nominal controls
    cov = data['cov']
    N = traj_true.shape[1]
    # t_GP = range(N)

    #plot realizations
    plt.plot(traj_true[1, :-1, :],traj_true[0, :-1, :],'--', color=c,linewidth=0.5)
    plt.plot(traj_true[1, -1, :],traj_true[0, -1, :],'--', color=c,linewidth=0.5, label=f'{labe}')

    # plot ellipsoids
    # print(cov.shape) 
    for i in range(0,N,2):
        mu = X_nom[0:2,i]
        Sigma = cov[0,i][0:2,0:2]
        D, V = np.linalg.eigh(Sigma)
        ellipse = V @ np.diag(np.sqrt(D*s)) @ circle
        ellipse += mu[:, None]
        plt.plot(ellipse[1, :], ellipse[0, :], color=c, linewidth=0.2)

    # plot reference
    plt.plot(X_nom[1,:],X_nom[0,:],'-', color='k', linewidth=1.0)

    coll= collision_percentage_GP(np.array(traj_true))
    print(traj, ':', coll)
################################################


# Start and goal markers
plt.plot(0.4,0, marker = "p",markersize=5, markerfacecolor='g', label='Start')
plt.plot(0.4,10,marker = "p",  markersize=5, markerfacecolor='r', label='Goal')

# Plot circular obstacle
obs_center = np.array([5, 0])
obs_radius = 1.2
theta = np.linspace(0, 2*np.pi, 100)
xunit = obs_radius * np.cos(theta) + obs_center[0]
yunit = obs_radius * np.sin(theta) + obs_center[1]
plt.fill( yunit,xunit, color=gray, edgecolor='k')

# Plot wall (rectangle from x=0 to 10, y=2 to 5)
plt.fill([2, 2, 5, 5],[-1, 11, 11, -1],  color=gray, edgecolor='k', alpha= 1)

plt.ylim([-1,11])
plt.xlim([-3,4])
   


plt.xlabel('Y')
plt.ylabel('X')
# plt.axis('equal')
plt.grid('True')
plt.tight_layout()
plt.legend(loc='upper left')
plt.show()

# plt.savefig("GP_LQR_GMM_U_CL.png", format="png", dpi=600)