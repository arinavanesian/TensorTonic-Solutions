import torch
def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = torch.tensor(float(x0), requires_grad=True)
    
    for _ in range(steps):
        # 1. Define the quadratic function
        # This creates the computational graph for the current step
        f_x = a * x**2 + b * x + c
        
        # 2. Compute the gradient f'(x) via backpropagation
        f_x.backward()
        
        # 3. Update x: x = x - lr * f'(x)
        # We use torch.no_grad() because we don't want the update math 
        # (subtraction/multiplication) to be part of the gradient history.
        with torch.no_grad():
            x -= lr * x.grad
            
        # 4. Zero the gradient for the next iteration
        # PyTorch accumulates (adds) gradients by default; clearing is mandatory
        x.grad.zero_()

    # Return as a standard Python float
    return float(x.item())