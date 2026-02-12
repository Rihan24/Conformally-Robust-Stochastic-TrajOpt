import os
import sys
import torch
import numpy as np

def get_controller_wrapper(controller_path):
    _controller = torch.load(controller_path, map_location=torch.device('cpu'), weights_only=False)
    _controller.cpu()

    def controller(x, xe, uref):
        u = _controller(torch.from_numpy(x).float().view(1,-1,1), torch.from_numpy(xe).float().view(1,-1,1), torch.from_numpy(uref).float().view(1,-1,1)).squeeze(0).detach().numpy()
        return u

    return controller

def get_system_wrapper(system):
    num_dim_x = system.num_dim_x
    num_dim_control = system.num_dim_control
    f_func = system.f_func
    B_func = system.B_func

    def f(x):
        dot_x = f_func(torch.from_numpy(x).float().view(1,-1,1)).detach().numpy()
        return dot_x.reshape(-1,1)

    def B(x):
        B_value = B_func(torch.from_numpy(x).float().view(1,-1,1)).squeeze(0).detach().numpy()
        return B_value

    def full_dynamics(x, u):
        return (f(x) + B(x).dot(u.reshape(-1,1))).squeeze(-1)

    return f, B, full_dynamics, num_dim_x, num_dim_control


def get_w_func_wrapper_from_checkpoint(checkpoint_path,w_lb=0.1, task="CAR"):
    """
    Load the saved model parameters and reconstruct W_func for evaluation
    
    Args:
        checkpoint_path: Path to the saved checkpoint file
        task: Task name (default "CAR")
    
    Returns:
        w_func_wrapper: Function that takes numpy array x and returns W matrix
    """
    import importlib
    import torch
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
    # checkpoint.cpu()
    # args = checkpoint['args']
    
    # Import the required modules
    config = importlib.import_module("config_" + task)
    system = importlib.import_module("system_" + task)
    model = importlib.import_module("model_" + task)
    
    num_dim_x = system.num_dim_x
    num_dim_control = system.num_dim_control
    
    # Recreate the model architecture
    model_W, model_Wbot, model_u_w1, model_u_w2, W_func, u_func = model.get_model(
        num_dim_x, num_dim_control, w_lb=w_lb, use_cuda=False
    )
    
    # Load the saved parameters
    model_W.load_state_dict(checkpoint['model_W'])
    model_Wbot.load_state_dict(checkpoint['model_Wbot'])
    
    # Set to evaluation mode
    model_W.eval()
    model_Wbot.eval()
    
    def w_func_wrapper(x):
        """
        Evaluate W_func for a given state x
        
        Args:
            x: numpy array of shape (n,) or (n,1) representing the state
            
        Returns:
            W: numpy array of shape (n,n) representing the W matrix
        """
        # Convert to torch tensor and ensure proper shape
        x_tensor = torch.from_numpy(x).float().view(1, -1, 1)
        
        # Evaluate W_func
        with torch.no_grad():
            W = W_func(x_tensor).squeeze(0).detach().numpy()
        
        return W
    
    return w_func_wrapper