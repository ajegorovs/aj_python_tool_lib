import torch

def conjugate_gradient_hess(Av, b, x0=None, max_iters = None, tol = 1e-10):
    x = torch.zeros_like(b) if x0 is None else x0  #init guess
    r           = b - Av(x)    # residual
    d           = r.clone()  # direction
    rr          = torch.dot(r,r)
    num_iters   = len(b) if max_iters is None else max_iters
    for _ in range(num_iters):
        Ad          = Av(d)
        step_size   = rr/ (d @ Ad)
        x           += step_size * d
        r           -= step_size * Ad
        rr_new      = torch.dot(r,r)
        if rr_new < tol: break
        d           = r + (rr_new/rr)*d
        rr          = rr_new.clone()
    return x.detach()