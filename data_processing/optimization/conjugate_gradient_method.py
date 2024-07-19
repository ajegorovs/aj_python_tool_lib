
import numpy as np


def conjugate_gradient(A, b, x0=None, max_iters = None, tol = 1e-10):
    x = np.zeros_like(b) if x0 is None else x0  #init guess
    r           = b - A @ x    # residual
    d           = r.copy()  # direction
    rr          = np.dot(r,r)
    num_iters   = len(b) if max_iters is None else max_iters
    for _ in range(num_iters):
        Ad          = A @ d
        step_size   = rr/ (d @ Ad)
        x           += step_size * d
        r           -= step_size * Ad
        rr_new      = np.dot(r,r)
        if rr_new < tol: break
        d           = r + (rr_new/rr)*d
        rr          = rr_new.copy()
    return x

    
if __name__ == "__main__":
    A = np.array([[3,2],[2,6]], dtype=float)
    b = np.array([2,-8], dtype=float)
    def f(x):
        return (0.5*x @ A @ x - np.dot(b,x))
    
    x0 = np.array([-1,4], dtype=float)
    x = conjugate_gradient(A,b,x0)
    print(f'{x = }; {f(x) = }; {A@ x - b}')
