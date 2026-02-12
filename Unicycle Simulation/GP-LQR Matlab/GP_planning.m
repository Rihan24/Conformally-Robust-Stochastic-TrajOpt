 import casadi.*
close all
%% Initialization
t_0 = 0;  
t_f = 10; 
% num_of_steps =50; % should be even
global dt;
dt = 0.2; % (t_f - t_0) / num_of_steps;
num_of_steps = (t_f - t_0) / dt;


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

%% Formulate covariance matrices

K_nom = load('K_array_right_N200.mat');

%covariance propagation
covariance = cell(1, length(tau));
covariance{1} = zeros(4);  % initial covariance

%% Manually enter estimated covariance
% width = 0.1;
% Q_noise = width^2/3* eye(4);  %width^2/3* eye(4);%0.01^2 * eye(4);
% [noise,Q_noise] =  zero_mean_gmm_noise(4);%0.01^2 * eye(4);  % process noise

%% Use MLE to estimate covariance
d = 4; Sample_sz = 1000;
true_mu = [0, 0, 0, 0];
true_sigma = 0.05^2*eye(d);
% uniform_noise = 0.2;
%%%%%%%%%%%%%%%%%%%%%%
a = 0.15;  %sqrt(3) *0.1;
low  = [-a/5; -a/5; -a; -a];
high = [ a/5;  a/5; a;  a];
low_mat  = repmat(low', Sample_sz, 1);  
high_mat = repmat(high', Sample_sz, 1); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
samples =   low_mat + (high_mat - low_mat).* rand(Sample_sz,d);     %zero_mean_gmm_noise(d,Sample_sz);   %low_mat + (high_mat - low_mat).* rand(Sample_sz,d);
% samples(:,1:2) = 1e-6;
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%zero_mean_gmm_noise(d,Sample_sz);
%-uniform_noise + 2*uniform_noise*rand(Sample_sz,d); 
% low + (high - low).* rand(4,1);
%mvnrnd(true_mu, true_sigma, Sample_sz);



[mean_est,cov_est]=MLE(samples);
disp(cov_est);
Q_noise=cov_est;
% Q_noise(1:2, :) = 1e-6;   % zero rows
% Q_noise(:, 1:2) = 1e-6;   % zero columns
% disp(Q_noise);


%% Propagate covariance
for i = 1 : num_of_steps
    %%CHECK
    [A,B]= GP_plan_linearize(X(:,i), U(:,i), dt);
    % Closed-loop A matrix
    A_cl = A - B * K_nom.K_array{i};

    % Propagate covariance
    covariance{i+1} = A_cl * covariance{i} * A_cl' + Q_noise;
end

%% Numerical integration and constraint to make zero gap
for k = 1 : num_of_steps % loop over control intervals
   opti.subject_to(X(:,k+1) == dyn_func(X(:,k),U(:,k))); % close the gaps
end

%% Constraints

%initial & final condition
opti.subject_to(X(:, 1) == [0 ; 0.4 ; 0;0]);
opti.subject_to(X(:, end) == [10; 0.4; 0;0]);

% State bounds
% CDF_inv= norminv(0.9, 0, 1);
p = 1- 0.1/2;
s = chi2inv(p, 2);
a_vec = [0;1;0;0];
for i = 1 : num_of_steps + 1
    opti.subject_to(X(2, i) + sqrt(s)*sqrt(a_vec'*covariance{i}(2,2)*a_vec) <= 2);
    dist = sqrt((X(1,i)-5)^2 + (X(2,i)-0)^2);
    n_vec = (1/dist)* [X(1,i)-5;X(2,i)-0;0;0];
    opti.subject_to( dist -1.2 - sqrt(s)* sqrt(n_vec'*covariance{i}*n_vec)  >= 0);
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
   
solution.cov = cell(1,length(tau));
for i=1:length(tau)
   solution.cov{i} = opti.value(covariance{i});
end

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
plot(x_vals, y_vals, 'b-', 'LineWidth', 1.5); hold on;

% Mark start and goal
plot(0, 0.4, 'go', 'MarkerSize', 5, 'MarkerFaceColor', 'g'); % start
plot(10, 0.4, 'ro', 'MarkerSize', 5, 'MarkerFaceColor', 'r'); % goal

% Plot obstacle (circular)
obs_center = [5, 0];
obs_radius = 1.2;
% theta = linspace(0, 2*pi, 100);
% plot(obs_center(1) + obs_radius*cos(theta), ...
%      obs_center(2) + obs_radius*sin(theta), 'k--', 'LineWidth', 1.5);
 
th = 0:pi/50:2*pi;
xunit = obs_radius * cos(th) + obs_center(1);
yunit = obs_radius * sin(th) + obs_center(2);
h = fill(xunit, yunit, gray );

% yline(2, 'r--', 'LineWidth', 1.5);  % horizontal line at y = 2
fill([0,10.5,10.5,0], [2,2,5,5], gray, 'EdgeColor', 'k');

view([90 -90])
% Formatting
xlabel('x');
ylabel('y');
% title('x-y Trajectory');
axis equal;
grid on;
% ylim([-4,4]);
% legend('Trajectory', 'Start', 'Goal', 'Obstacle');


for k = 1:length(tau)
    mu = solution.X(1:2, k);  % mean [x; y]
    Sigma = solution.cov{k}(1:2, 1:2);  % 2x2 position covariance

    % Get ellipse points
    [V, D] = eig(Sigma);
    theta = linspace(0, 2*pi, 100);
    circle = [cos(theta); sin(theta)];
    ellipse = V * sqrt(D * s) * circle;

    % Shift to mean
    ellipse = ellipse + mu;

    % Plot
    plot(ellipse(1, :), ellipse(2, :), 'r', 'LineWidth', 1);
end

% figure;
% plot(tau,solution.U);

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

