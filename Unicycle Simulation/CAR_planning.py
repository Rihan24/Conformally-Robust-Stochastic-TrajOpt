import casadi as ca
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from np2pth import get_system_wrapper, get_controller_wrapper, get_w_func_wrapper_from_checkpoint
import importlib
import sys
sys.path.append('systems')
sys.path.append('configs')
sys.path.append('models')

# Parameters
a1 = 0.1
a2 = 10

# Time settings
t_0 = 0
t_f = 10
dt = 0.05 #(t_f - t_0) / num_of_steps
num_of_steps =  int((t_f - t_0)/dt)
print("dt:", dt)

tau = np.linspace(t_0, t_f, num_of_steps + 1)
print(len(tau))

# Symbols
x = ca.SX.sym('x', 4)
u = ca.SX.sym('u', 2)

# Dynamics
def dynamicsFunc(x, u):
    x_dot = ca.vertcat(x[3]*ca.cos(x[2]),
                       x[3]*ca.sin(x[2]),
                       u[0],
                       u[1])
    return x + x_dot * dt

# Cost
def stageCost(x, u):
    Q = 0.0 * np.eye(4)
    R = 0.1 * np.eye(2)
    return ca.mtimes([x.T, Q, x]) + ca.mtimes([u.T, R, u])

dyn_func = ca.Function('dyn_func', [x, u], [dynamicsFunc(x, u)])
lag_cost = ca.Function('lag_cost', [x, u], [stageCost(x, u)])

# Optimization problem
opti = ca.Opti()

X = opti.variable(4, len(tau))
U = opti.variable(2, len(tau))

# Cost function
J = 0
for i in range(num_of_steps):
    J += lag_cost(X[:, i], U[:, i])
opti.minimize(J)

# Dynamics constraints
for k in range(num_of_steps):
    opti.subject_to(X[:, k + 1] == dyn_func(X[:, k], U[:, k]))

# Boundary conditions
opti.subject_to(X[:, 0] == ca.vertcat(0, 0.4, 0, 0))
opti.subject_to(X[:, -1] == ca.vertcat(10, 0.4, 0, 0))

# Obstacle constraints
obs_radius = 1.2
# q_data = scipy.io.loadmat('../uncertainty_quantification/q_g0.7_N200_K2k_W0.5.mat')
# q = q_data['q'].flatten()  # Assumes q is (N,) shape


# for i in range(num_of_steps):
#     rad_safety = np.sqrt(np.sqrt(m_ub / 1) * 0.2361)  *np.sqrt(0.132)               #np.sqrt(np.sqrt(m_ub / m_lb) * q[i])
#     opti.subject_to(X[1, i] <= 2 - rad_safety)
#     dist = ca.sqrt((X[0, i] - 5)**2 + (X[1, i])**2)
#     opti.subject_to(dist >= obs_radius + rad_safety)

# CP constraint tightening

system = importlib.import_module('system_CAR')
f, B, _, num_dim_x, num_dim_control = get_system_wrapper(system)
w_func = get_w_func_wrapper_from_checkpoint(checkpoint_path='log_CAR_0.5_10_1/model_best.pth.tar', w_lb=0.5, task = "CAR")
M = w_func(np.array([0, 0.4, 0, 0]))
s=0.167
for i in range(num_of_steps):
    a_vec = ca.vertcat(0,1, 0, 0)       
    opti.subject_to(X[1, i] + s*ca.sqrt(ca.mtimes([a_vec.T, M, a_vec]))<= 2 )

    dist = ca.sqrt((X[0, i] - 5)**2 + (X[1, i])**2)
    n_vec = ca.vertcat(X[0,i] - 5, X[1,i] - 0, 0, 0) / dist
    opti.subject_to(dist >= obs_radius + +s *ca.sqrt(ca.mtimes([n_vec.T, M, n_vec]))) 

# Control bounds

for i in range(num_of_steps + 1):
    opti.subject_to(opti.bounded(-1, U[0, i], 1))
    opti.subject_to(opti.bounded(-10, U[1, i], 10))

# L = 5
# for i in range(num_of_steps + 1):
#     opti.subject_to(opti.bounded(-10 + L*s, U[0, i], 10- L*s))
#     opti.subject_to(opti.bounded(-10 + L*s, U[1, i], 10- L*s))


#Initial guess using straight interpolation
for i in range(num_of_steps + 1):
    alpha = i / num_of_steps
    x_init = (1 - alpha) * np.array([0, 0.4, 0, 0]) + alpha * np.array([10, 0.4, 0, 0])
    opti.set_initial(X[:, i], x_init)

################################ Initial guess (double linear interpolation)
# P0 = np.array([0.4, 10])
# P1 = np.array([5, -4])
# P2 = np.array([10, 0.4])
# theta_init = 0
# v_init = 1
# x0 = np.concatenate([P0, [theta_init, v_init]])
# x1 = np.concatenate([P1, [theta_init, v_init]])
# xf = np.concatenate([P2, [theta_init, v_init]])

# N1 = (num_of_steps + 1) // 2
# N2 = (num_of_steps + 1) - N1

# for i in range(num_of_steps + 1):
#     if i < N1:
#         alpha = i / (N1 - 1)
#         x_init = (1 - alpha) * x0 + alpha * x1
#     else:
#         alpha = (i - N1) / (N2 - 1)
#         x_init = (1 - alpha) * x1 + alpha * xf
#     opti.set_initial(X[:, i], x_init)
########################################################

opti.set_initial(U, np.ones((2, num_of_steps + 1)))

# Solver setup
opti.solver('ipopt')
print("Starting to solve...")
import time
start_time = time.time()
sol = opti.solve()
time_taken = time.time() - start_time
print("Solving finished in", time_taken, "seconds")


#SAVE TRAJECTORY
# np.savez('CAR_ref_U0.1_smaller.npz', X=sol.value(X), U=sol.value(U))

# Store solution
solution = {
    'output': sol,
    'X': opti.value(X),
    'U': opti.value(U),
    'tau': tau,
    'cost': opti.value(J),
    'NLP_time_taken': time_taken
}

# Plotting
x_vals = solution['X'][0, :]
y_vals = solution['X'][1, :]

plt.figure()
plt.plot(y_vals, x_vals, 'b', label='Trajectory')
plt.plot(0.4, 0, 'go', label='Start')
plt.plot(0.4, 10, 'ro', label='Goal')

# Obstacle
obs_center = [5, 0]
th = np.linspace(0, 2 * np.pi, 100)
xunit = obs_radius * np.cos(th) + obs_center[0]
yunit = obs_radius * np.sin(th) + obs_center[1]
plt.fill(yunit, xunit, color='gray')

# Upper constraint zone
plt.fill([2, 2, 5, 5],[0, 10, 10, 0], color='gray', edgecolor='k')

theta = np.linspace(0, 2*np.pi, 100)
circle = np.vstack((np.cos(theta), np.sin(theta)))  # shape (2, 100)
for i in range(0,num_of_steps,2):
    # fac= np.sqrt(1/w_lb)* q + delta_v*((1-gamma**i)/(1-gamma))
    mu = np.array([x_vals[i],y_vals[i]])
    # Sigma = np.array(w_func(xstar[i,:,0]))[:2,:2]
    Sigma = np.array(w_func(solution['X'][:, 0]))[:2,:2]
    # Sigma = np.eye(2)
    # dist = np.sqrt((xstar[i,0,0]-5)**2 + (xstar[i,1,0]-0)**2 )
    # n_vec = (1 - 1.2/dist)*(1/(dist-1.2))* np.array([np.array(xstar[i,0,0])-5,np.array(xstar[i,1,0])-0,0,0]);


    # Eigen decomposition
    D, V = np.linalg.eigh(s**2 *Sigma)
    ellipse = V @ np.diag(np.sqrt(D)) @ circle
    ellipse += mu[:, None]

    # Plot ellipsoid
    plt.plot(ellipse[1, :], ellipse[0, :], 'r', linewidth=0.5)

plt.xlabel('y')
plt.ylabel('x')
plt.xlim([-4,6])
plt.axis('equal')
# plt.grid(True)
# plt.legend()
# plt.title('x-y Trajectory')
plt.show()
