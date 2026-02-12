import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ----------------------
# Parameters and Setup
# ----------------------
g = 9.81
pi=3.141
t_0 = 0
t_f = 5.0
num_of_steps = 200
dt = (t_f - t_0) / num_of_steps
tau = np.linspace(t_0, t_f, num_of_steps + 1)

# ----------------------
# Symbolic Variables
# ----------------------
x = ca.SX.sym('x', 10)  # State: px, py, pz, vx, vy, vz, f, phi, theta, psi
u = ca.SX.sym('u', 4)   # Control: dot(f), dot(phi), dot(theta), dot(psi)

# ----------------------
# Dynamics Function
# ----------------------
def dynamicsFunc(x, u, dt, g=9.81):
    vx = x[3]
    vy = x[4]
    vz = x[5]
    f  = x[6]
    phi = x[7]
    theta = x[8]

    x_dot = ca.vertcat(
        vx,
        vy,
        vz,
        -f * ca.sin(theta),
        f * ca.cos(theta) * ca.sin(phi),
        g - f * ca.cos(theta) * ca.cos(phi),
        u[0],  # dot(f)
        u[1],  # dot(phi)
        u[2],  # dot(theta)
        u[3]   # dot(psi)
    )
    return x + dt * x_dot

# ----------------------
# Stage Cost Function
# ----------------------
def stageCost(x, u):
    Q = 0.0 * np.eye(10)
    R = 0.01 * np.eye(4)
    return ca.mtimes([x.T, Q, x]) + ca.mtimes([u.T, R, u])


def plot_sphere(ax, center, radius, color='gray', alpha=1):
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, edgecolor='gray')

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

# ----------------------
# CasADi Function Wrappers
# ----------------------
dyn_func = ca.Function('dyn_func', [x, u], [dynamicsFunc(x, u, dt)])
lag_cost = ca.Function('lag_cost', [x, u], [stageCost(x, u)])

# ----------------------
# Optimization Problem Setup
# ----------------------
opti = ca.Opti()
X = opti.variable(10, len(tau))
U = opti.variable(4, len(tau))

# Objective function
J = 0
for i in range(num_of_steps):
    J += lag_cost(X[:, i], U[:, i])
opti.minimize(J)

# Dynamics constraints
for k in range(num_of_steps):
    opti.subject_to(X[:, k + 1] == dyn_func(X[:, k], U[:, k]))

# Initial and final conditions
x0 = [0, 0, 0.5, 0, 0, 0, g, 0, 0, 0]
xf = [2, 2, 0.5, 0, 0, 0, g, 0, 0, 0]
opti.subject_to(X[:, 0] == x0)
opti.subject_to(X[:, -1] == xf)


# STATE BOUNDS
for i in range(num_of_steps):
    opti.subject_to(opti.bounded(0.5*g,X[6,i],2*g))
    opti.subject_to(opti.bounded(-pi/3,X[7,i],pi/3))
    opti.subject_to(opti.bounded(-pi/3,X[8,i],pi/3))

# Control bounds
for i in range(num_of_steps + 1):
    opti.subject_to(opti.bounded(-1, U[0, i], 1))  # dot(f)
    opti.subject_to(opti.bounded(-1, U[1, i], 1))    # dot(phi)
    opti.subject_to(opti.bounded(-1, U[2, i], 1))    # dot(theta)
    opti.subject_to(opti.bounded(-1, U[3, i], 1))    # dot(psi)

# OBSTACLE AVOIDANCE BOUNDS

obstacles = [
    {'center': np.array([0.8, 1.0, 0.5]), 'radius': 0.3},
    {'center': np.array([1.3, 1.4, 0.5]), 'radius': 0.3},
    {'center': np.array([1.7, 0.8, 0.5]), 'radius': 0.3}
]
for obs in obstacles:
    center = obs['center']
    radius = obs['radius']
    for i in range(num_of_steps + 1):
        dist = ca.norm_2(X[0:3, i] - center)
        opti.subject_to(dist >= radius+0.1 )  # 0.1 m safety margin

# Initial Guess
for i in range(num_of_steps + 1):
    alpha = i / num_of_steps
    x_init = (1 - alpha) * np.array(x0) + alpha * np.array(xf)
    opti.set_initial(X[:, i], x_init)
opti.set_initial(U, np.zeros((4, num_of_steps + 1)))

# Solver settings
opti.solver('ipopt')
print("Solving...")

start_time = time.time()
sol = opti.solve()
print("Solved in", time.time() - start_time, "seconds")

# ----------------------
# Extract and Plot Results
# ----------------------
X_val = sol.value(X)
U_val = sol.value(U)

# np.savez('quad1.npz', X=X_val, U=U_val)

## READING ABOVE DATA
# data = np.load('quad_traj_data.npz')
# X_val = data['X']
# U_val = data['U']

# Plot trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X_val[0, :], X_val[1, :], X_val[2, :], 'b', label='Trajectory')
ax.scatter(x0[0], x0[1], x0[2], color='green', label='Start')
ax.scatter(xf[0], xf[1], xf[2], color='red', label='Goal')

for obs in obstacles:
    plot_sphere(ax, obs['center'], obs['radius'], color='gray', alpha=0.5)


set_axes_equal(ax)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Quadrotor Trajectory with Obstacles')
ax.legend()
ax.set_box_aspect([1, 1, 1]) 
plt.show()


fig = plt.figure()
# plt.plot(tau,X_val[6])
plt.plot(tau,X_val[7])
plt.plot(tau,X_val[8])

plt.show()



