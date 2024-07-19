import numpy as np

def line_search(f, f_grad, x0, d, max_iters, beta = 1e-4, step_size = 1, rho = 0.5, debug = False):
    # armijo: f_test <= f_0 + beta*step_size*dir_derivative
    alpha = step_size
    f_0 = f(*x0)
    gr = f_grad(*x0)
    grad_f_d = np.dot(gr.T, d).item()
    for iter in range(max_iters):
        x_test = x0 + alpha*d
        f_test = f(*x_test)
        f_expected = f_0 + beta*alpha*grad_f_d
        if f_test <= f_expected:
            break
        alpha *= rho
        if debug: print(f'{iter = }; {alpha = }')
    return alpha, x_test

if __name__ == "__main__":
    from sympy import Matrix, lambdify
    from sympy.abc import x, y, a 
    import matplotlib.pyplot as plt

    vars = [x,y]
    f0      = x**2 + x*y + y**2
    f       = lambdify(vars, f0)
    d,x0    = np.array([-1,-1]), np.array([1,2])
    lx,ly   = x0 + a*d 
    grad_f  = lambdify(vars, Matrix([f0]).jacobian(vars).T)
    alpha   = 10
    rho     = 0.75
    beta    = 1e-4
    step_size = 10
    alpha, x_sol = line_search(f, grad_f, x0, d, max_iters = 20, step_size =step_size, rho = rho, debug=1)

    f_of_a  = f0.subs({x: lx, y: ly})
    f_of_an = lambdify(a, f_of_a)
    nm      = 300
    xss     = np.linspace(0, step_size, nm)
    plt.plot(xss, f_of_an(xss), label= r'$\hat{f}(\alpha)$')
    plt.scatter([alpha],f_of_an(alpha))
    plt.show()