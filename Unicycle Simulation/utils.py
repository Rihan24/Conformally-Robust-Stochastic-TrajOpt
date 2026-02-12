import contextlib
import numpy as np
from scipy.stats import multivariate_normal
params = np.load("../Crazyflie Hardware/CrazyflieNoise_params.npz")
mu_hat = params["mu_hat"]
Sigma_hat = params["Sigma_hat"]

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def EulerIntegrate(controller, f, B, xstar, ustar, xinit,w_func, t_max = 10, dt = 0.05, gamma=0.1, with_tracking = False, sigma = 0., noise_bound = None):
    t = np.arange(0, t_max, dt)

    trace = []
    u = []

    xcurr = xinit
    # print('xinit :',xinit)
    trace.append(xcurr)

    D = np.diag([1, 1, 1, 1])   #FOR CAR
    # D = np.diag([1, 1, 1, 1,1,1,1,1,1])   #FOR QUADROTOR
    fro_norm = np.linalg.norm(D, 'fro')
    omega_CP_sumterm = 0
    omega_CP_array=[]

    a = sigma
    low = np.array([
        -np.sqrt(3) * a / 5,  # dim 1
        -np.sqrt(3) * a / 5,  # dim 2
        -np.sqrt(3) * a,      # dim 3
        -np.sqrt(3) * a          # dim 4
    ])
    high = np.array([
        np.sqrt(3) * a / 5,   # dim 1
        np.sqrt(3) * a / 5,   # dim 2
        np.sqrt(3) * a,      # dim 3
        np.sqrt(3)*a            # dim 4
    ])
    # noise = np.random.uniform(low, high).reshape(4, 1) 
    # print(D@noise)
    
    for i in range(len(t)):
        if with_tracking:
            xe = xcurr - xstar[i]
        # print('xcurr :',xcurr)
        # print('xe :',xe)
        # print('ustar :',ustar[i])
        ui = controller(xcurr, xe, ustar[i]) if with_tracking else ustar[i]
        # print("xcurr :",xcurr)
        # print("xe :",xe)
        # print("ui :",ui)


        ################################################## NON_ISOTROPIC NOISE
        # cov = np.diag([(sigma/5)**2, (sigma/5)**2, sigma**2, sigma**2])
        # noise = np.random.multivariate_normal(mean=np.zeros(4), cov=cov).reshape((4, 1))

        ################################################### GAUSSIAN or GAUSSIAN MIXTURE NOISE
        # if sigma==0.0:
        #     noise = zero_mean_gmm_noise(4, N=1).T 
        # else: 
        #     noise = np.random.multivariate_normal(mean=np.zeros(4), cov=cov).reshape((4, 1))


        # print(noise)
        #zero_mean_gmm_noise(4, N=1).T    #GMM        
        # np.random.randn(*xcurr.shape) * sigma   # isotropic gaussian

        # Loading MLE file
        # noise = np.random.multivariate_normal(mean=mu_hat, cov=Sigma_hat).reshape(-1,1) 
        # print('noise :',noise)

        ###################################################UNIFORM NOISE        
        noise =np.random.uniform(low, high).reshape(4, 1)   #np.random.uniform(low, high).reshape(4, 1)   #np.random.uniform(-sigma, sigma, size=(4,1))

        ################################################### CAUCHY NOISE
        # noise = cauchy_noise(n=4).T
        ################################################### If not noise_bound:
        # noise_bound = 3 * sigma
        # noise[noise>noise_bound] = noise_bound
        # noise[noise<-noise_bound] = -noise_bound

        #################################################### ONLY TO V AND A
        # noise[0:2] = 0.0
        # print(noise)

        # print(D@noise)
        dx = f(xcurr) + B(xcurr).dot(ui) 
        xnext =  xcurr + dx*dt  + D@noise
        # xnext[xnext>100] = 100
        # xnext[xnext<-100] = -100

        trace.append(xnext)
        u.append(ui)
        xcurr = xnext


        # CP terms
        M_curr = np.linalg.inv(w_func(xstar[0]))     
        omega_CP_sumterm+=  np.sqrt((D@noise).T @ M_curr @ (D@noise)).item()*(gamma**i) 
        # omega_CP_array.append(omega_CP_sumterm) 

        #np.sqrt((noise).T @ M_curr @ (noise)).item()*(gamma**i)    
        #np.sqrt((D@noise).T @ M_curr @ (D@noise)).item()*(gamma**i)   
        #(np.linalg.norm(D@noise)*(gamma**i))
    

    return trace, u,  omega_CP_sumterm    #fro_norm*omega_CP_sumterm






def zero_mean_gmm_noise(d, N=1):
    """
    Generate N samples from a d-dimensional zero-mean Gaussian mixture.

    Returns:
        samples: (N x d) array where each row is a sample
        Sigma_mix: (d x d) covariance matrix of the mixture
    """
    # Define mixture components
    K = 3
    weights = np.array([0.1, 0.8, 0.1])

    # Component means (overall mean must be zero)
    mu = np.zeros((K, d))
    mu[0, :] = -0.5
    mu[1, :] = 0.0
    mu[2, :] = 0.5

    # Check zero-mean
    mix_mean = np.sum(weights[:, None] * mu, axis=0)
    assert np.linalg.norm(mix_mean) < 1e-10, 'Mixture is not zero mean!'

    # Component covariances
    Sigma = np.zeros((K, d, d))
    Sigma[1] = (0.05 ** 2) * np.eye(d)
    Sigma[0] = (0.05 ** 2) * np.eye(d)
    Sigma[2] = (0.05**2) * np.eye(d)
    

    # Moment-matched covariance
    Sigma_mix = np.zeros((d, d))
    for k in range(K):
        mu_k = mu[k][:, None]  # make it a column vector
        Sigma_mix += weights[k] * (Sigma[k] + mu_k @ mu_k.T)

    # Sample from the GMM
    samples = np.zeros((N, d))
    component_choices = np.random.choice(K, size=N, p=weights)
    for i in range(N):
        k = component_choices[i]
        samples[i, :] = np.random.multivariate_normal(mean=mu[k], cov=Sigma[k])

    return samples

def cauchy_noise(n, N=1, loc=0.0, scale=0.001):
    """
    Generate N samples of n-dimensional Cauchy noise.

    Parameters:
    - n: Dimension of each sample
    - N: Number of samples
    - loc: Location (median)
    - scale: Scale (like standard deviation but for Cauchy)

    Returns:
    - noise: N x n array of Cauchy noise samples
    """
    return np.random.standard_cauchy((N, n)) * scale + loc

# Example: Generate 100 samples of 5D Cauchy noise


