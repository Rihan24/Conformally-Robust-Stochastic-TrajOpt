import casadi.*
close all



%% Initialization
t_0 = 0;  
t_f = 10;  

% num_of_steps = 200; % should be even
global dt;
dt = 0.05; % (t_f - t_0) / num_of_steps;
num_of_steps = (t_f - t_0) / dt;
disp(num_of_steps);

tau = linspace(t_0, t_f, num_of_steps + 1);

% State vector
x = casadi.SX.sym('x', 4);

% Control vector
u = casadi.SX.sym('u', 2);

% Time variable
t = casadi.SX.sym('t');

% Model equations
dx = dynamicsFunc(x,u);

% Objective term
L = stageCost(x,u);

% Continuous time dynamics
dyn_func = casadi.Function( 'dyn_func', {x, u}, {dx}, {'x', 'u'}, {'xdot'});
lag_cost = casadi.Function( 'lag_cost', {x, u}, {L}, {'x', 'u'}, {'Stage cost'});

%% Set up solver and declare variables
% Start with an empty NLP
opti = casadi.Opti();

% All states and control variables (inclusive of redundant ones)
X = opti.variable(4, length(tau));
% X.set_initial(zeros(problem.nx, len(tau)))

U = opti.variable(2, length(tau));
% U.set_initial(zeros(problem.nu, len(tau)))

%! Check the size of U (no control required at tf)

%% Formulate NLP


J = 0;
for i = 1 : num_of_steps
        J = J + lag_cost(X(:,i), U(:,i));
end


%% Numerical integration and constraint to make zero gap
for k = 1 : num_of_steps % loop over control intervals
   opti.subject_to(X(:,k+1) == dyn_func(X(:,k),U(:,k))); % close the gaps
end

%% Constraints

%initial & final condition
opti.subject_to(X(:, 1) == [0 ; 0.4 ; 0;0.1]);
opti.subject_to(X(:, end) == [10; 0.4; 0;0]);
% opti.subject_to(X(1, end) == 10);
% opti.subject_to(X(2, end) == 0.4);
% opti.subject_to(X(4, end) == 0);
% State bounds

obs_radius = 1.2;
q= 0.01;  %load('..\uncertainty_quantification\q_g0.689_N200_K2k_W0.1.mat');
w_lb=1;w_ub=2;
W_0= [ 1.3579274e+00 -3.5383552e-03  1.5607670e-02 -3.7802804e-01; -3.5383552e-03  1.3648285e+00 -2.9831091e-01 -4.8092008e-04; 1.5607670e-02 -2.9831091e-01  1.2622755e+00 -1.6468447e-02;-3.7802804e-01 -4.8092008e-04 -1.6468447e-02  1.4045478e+00];
% M_0 = [0.1269, -0.00205,-0.005083,-0.02929;-0.00205,0.132,-0.03841,0.00046;-0.005083,-0.03841,0.1539,0.01215;-0.02929,0.0004668,0.01215,0.13595816];  %CN 2
% M_0= [1.4384872   0.38096216 -0.15466821 -0.6944782;  0.38096216  1.427939   -0.7532662  -0.05620122; -0.15466821 -0.7532662   0.839962    0.22653538; -0.6944782  -0.05620122  0.22653538  0.68573666];
a_vec = [0;1; 0; 0];
gamma=0.2;delta_v= 0.01;
for i = 1 : num_of_steps 
    fac = sqrt(1/w_lb)*q + delta_v*((1-gamma^i)/(1-gamma));
    opti.subject_to(X(2, i) <= 2 - fac*sqrt(a_vec'*W_0*a_vec));
    dist = sqrt((X(1,i)-5)^2 + (X(2,i)-0)^2);
    n_vec = (1/dist)* [X(1,i)-5;X(2,i)-0;0;0];
    opti.subject_to( dist >= obs_radius + fac*sqrt(n_vec'*W_0*n_vec));
end

% Control action bounds
for i = 1 : num_of_steps + 1
    opti.subject_to(U(1, i) >= -1);
    opti.subject_to(U(1, i) <= 1);
    opti.subject_to(U(2, i) <= 1);
    opti.subject_to(U(2, i) >= -1);
end

%% Optimization solver
for i = 1 : num_of_steps + 1
    alpha = (i-1)/num_of_steps;
    x0 = [0;0.4;0;1];
    xf = [10;0.4;0;1];
    opti.set_initial(X(:,i), (1-alpha)*x0 + alpha*xf);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% P0 = [0.4; 10];P1 = [5; -4];P2 = [10; 0.4];
% % Split steps (you can adjust the ratio)
% N1 = floor((num_of_steps + 1) / 2);N2 = (num_of_steps + 1) - N1;
% % Initial and final full states (assuming same theta and v for simplicity)
% theta_init = 0; v_init = 1;x0 = [P0; theta_init; v_init];x1 = [P1; theta_init; v_init];xf = [P2; theta_init; v_init];
% for i = 1 : num_of_steps + 1
%     if i <= N1
%         alpha = (i - 1) / (N1 - 1);
%         x_init = (1 - alpha) * x0 + alpha * x1;
%     else
%         alpha = (i - N1) / (N2 - 1);
%         x_init = (1 - alpha) * x1 + alpha * xf;
%     end
%     opti.set_initial(X(:,i), x_init);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

opti.set_initial(U, zeros(2, num_of_steps+1));
opti.minimize(J)  % minimise the objective function

% NLP solver used here is ipopt
% opts = struct('ipopt',options.ipopt);
opti.solver('ipopt'); %, opts)  % backend NLP solver

disp('Starting to solve...')
tic();
sol = opti.solve();  % Solve the actual problem
solution.NLP_time_taken = toc();
disp('Solving finished!')

%% Store to solution
solution.output = sol; % Store CasADi output
solution.X = opti.value(X);
% solution.U_app = U_app;
solution.tau=value(tau);
solution.cost = opti.value(J);
solution.U=opti.value(U);
solution.W_0=W_0;
solution.q=q;
x_star=opti.value(X);
u_star=opti.value(U);
% save('x_star_GMM0.1.mat', 'x_star');
% save('u_star_GMM0.1.mat', 'u_star');

%% Plots 
orange = [0.8500 0.3250 0.0980];
yellow= [0.9290 0.6940 0.1250];
purple = [0.4940 0.1840 0.5560];
olive_green = [0.4660 0.8740 0.1880];
maroon= [0.6350 0.0780 0.1840];
gray = [.7 .7 .7];
bad_blue=[0 0.4470 0.7410];

x_vals = solution.X(1, :);  % x-position
y_vals = solution.X(2, :);  % y-position
figure;
plot(x_vals, y_vals, 'b.', 'LineWidth', 1.5); hold on;

% Mark start and goal
plot(0, 0.4, 'go', 'MarkerSize', 5, 'MarkerFaceColor', 'g', 'DisplayName','Start'); % start
% legend(gca,'show');
plot(10, 0.4, 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r'); % goal

% Plot obstacle (circular)
obs_center = [5, 0];

% theta = linspace(0, 2*pi, 100);
% plot(obs_center(1) + obs_radius*cos(theta), ...
%      obs_center(2) + obs_radius*sin(theta), 'k--', 'LineWidth', 1.5);
 
th = 0:pi/50:2*pi;
xunit = obs_radius * cos(th) + obs_center(1);
yunit = obs_radius * sin(th) + obs_center(2);
h = fill(xunit, yunit, gray );

% yline(2, 'r--', 'LineWidth', 1.5);  % horizontal line at y = 2
fill([0,10,10,0], [2,2,5,5], gray, 'EdgeColor', 'k');


for k = 1:length(tau)-1
    mu = solution.X(1:2, k);  % mean [x; y]
    Sigma = W_0(1:2, 1:2); 
    fac = sqrt(1/w_lb)*q  + delta_v*((1-gamma^k)/(1-gamma));
    
    [V, D] = eig(Sigma);
    theta = linspace(0, 2*pi, 100);
    circle = [cos(theta); sin(theta)];
    ellipse = V * sqrt(D * fac^2) * circle;
    ellipse = ellipse + mu;
    
    plot(ellipse(1, :), ellipse(2, :), 'r', 'LineWidth', 0.2);
end

view([90 -90])
% Formatting
xlabel('x');
ylabel('y');
% title('x-y Trajectory');
axis equal;
grid on;
% ylim([-4,4]);
% legend('Trajectory', 'Start', 'Goal', 'Obstacle');

% figure;
% hold on;
% plot(tau,solution.X(4,:));
% plot(tau,solution.X(2,:));
% hold off;

%% Function Definitions
function x_next = dynamicsFunc(x, u)
    global dt;
    % Continuous dynamics model
%     x_dot = [x(2); 
%              cos(x(5)) * u(1); 
%              x(4); 
%              sin(x(5)) * u(1); 
%              x(6); 
%              u(2)];
    x_dot=[x(4)*cos(x(3));x(4)*sin(x(3));u(1);u(2)];
    % Euler integration to get next state
    x_next = x + x_dot * dt;
end

function L = stageCost(x, u)
    % LQR cost: x'*Q*x + u'*R*u
    Q = 0.0*eye(4,4);          % 6x6 zero matrix
    R = 0.1*eye(2);            % 2x2 identity matrix

%     L = x.' * Q * x + u.' * R * u; 
    L= dot(x, Q*x) + dot(u, R*u);
end

