import numpy as np


class Newton():
    def __init__(self, print_func=print, ls='wolfe_armijo'):
        self.linesearch = getattr(self, ls)
        self.exit_flag = False
        # if verb:
        #     self.print = lambda str: print(str)
        # else:
        #     self.print = lambda std: None
        # self.verb = verb
        self.print = print_func
        self.cost = {
            'objf': 0,
            'grads': 0
        }
        # self.H = []

    def run_vec(self, fun, grad_fun, hess_fun, x0, step=1., iters=1000, tol=1e-10):
        self.dim = len(x0)
        X = [x0]
        grad = [grad_fun(X[-1])]
        for k in range(iters):
            # hh = hess_fun(X[-1])
            d = -np.linalg.solve(hess_fun(X[-1]), grad[-1])
            if np.linalg.norm(d) < tol:
                self.print('Tolerance achieved')
                break
            ang = cos_vec(d, grad[-1])
            self.print(f'cos d and grad:{ang}')
            if ang > -0.05:
                X_cand = self.linesearch(X[-1], -grad[-1], fun, grad_fun, step)
            else:
                X_cand = self.linesearch(X[-1], d, fun, grad_fun, 1.)
                if self.exit_flag:
                    self.exit_flag = False
                    X_cand = self.linesearch(X[-1], -grad[-1], fun, grad_fun, step)
            if not self.exit_flag:
                X.append(X_cand)
                grad.append(grad_fun(X[-1]))
                self.cost['grads'] += 1
            else:
                self.print('Line search failed')
                break
        return X[-1]

    def run_matrix(self, fun, grad_fun, hess_inv_action, x0, step=1.,
                   iters=1000, tol=1e-10):
        self.dim = len(x0)
        self.x = [x0]
        grad = [grad_fun(self.x[-1])]
        for k in range(iters):
            # hh = hess_fun(self.x[-1])
            # if np.linalg.norm(d) < tol:
            if np.linalg.norm(grad[-1]) < tol:
                self.print('Tolerance achieved')
                break
            # d = -hess_inv_action(grad[-1], self.x[-1])
            d = -hess_inv_action(self.x[-1], grad[-1])  # Changed convention!
            self.print(f'grad. norm: {np.linalg.norm(grad[-1])}, '
                       f' d norm: {np.linalg.norm(d)}')
            ang = cos_vec(d, grad[-1])
            self.print(f'cos d and grad:{ang}')
            if ang > -1e-2:
                x_cand = self.linesearch(self.x[-1], -grad[-1], fun, grad_fun, step)
            else:
                x_cand = self.linesearch(self.x[-1], d, fun, grad_fun, step)
                if self.exit_flag:
                    self.exit_flag = False
                    x_cand = self.linesearch(self.x[-1], -grad[-1], fun, grad_fun, step)
            if not self.exit_flag:
                self.x.append(x_cand)
                grad.append(grad_fun(self.x[-1]))
                self.cost['grads'] += 1
            else:
                self.print('Line search failed')
                break
        ff = fun(self.x[-1])
        if np.isnan(ff) or np.isinf(ff):
            self.print(f'Error: objective function is {ff} at candidate point.')
        return self.x[-1]

    def wolfe_armijo(self, x, d, fun, grad_fun, step):
        grad0 = grad_fun(x)
        f0 = fun(x)
        self.cost['grads'] += 1
        self.cost['objf'] += 1
        step_min = 1e-20
        step_max = np.nan

        inner0 = (grad0*d).sum()

        c1 = 0.0001
        c2 = 0.95
        for t in range(100):
            x1 = x + step*d
            grad1 = grad_fun(x1)
            f1 = fun(x1)
            self.cost['grads'] += 1
            self.cost['objf'] += 1
            wolfe = -(grad1*d).sum() < -c2*inner0
            armijo = f1 < f0 + c1*step*inner0
            self.print(f'{wolfe}, {armijo}')
            if wolfe and armijo:
                self.print(f'Accepted step: {step}. Decreased objf from {f0} to {f1}')
                norm0 = np.linalg.norm(grad0)
                norm1 = np.linalg.norm(grad1)
                self.print(f'Accepted step: {step}. Decreased grad. norm from {norm0} to {norm1}')
                return x1
            elif (not wolfe) and (not armijo):
                self.exit_flag = True
                self.print('Error')
                return False
            elif (not wolfe) and armijo:
                step_min = np.copy(step)
                if np.isnan(step_max):
                    step = 2*step
                else:
                    step = (step_min + step_max)/2
                self.print(f'Increase to {step}')
            elif wolfe and (not armijo):
                step_max = np.copy(step)
                self.print(f'Decrease to {step}')
                step = (step_min + step_max)/2
        self.exit_flag = True
        self.print('100 steps')
        return False

    def backtrack_armijo(self, x, d, fun, grad_fun, step):
        grad0 = grad_fun(x)
        f0 = fun(x)
        self.cost['grads'] += 1
        self.cost['objf'] += 1
        # step_min = 1e-20
        c1 = 0.0001
        inner = np.sum(grad0 * d)
        for t in range(100):
            x1 = x + step*d
            f1 = fun(x1)
            self.cost['objf'] += 1
            armijo = f1 < f0 + c1*step*inner
            if armijo:
                self.print(f'Accepted step: {step}. Decreased objf from {f0} to {f1}')
                norm0 = np.linalg.norm(grad0)
                grad1 = grad_fun(x1)
                norm1 = np.linalg.norm(grad1)
                self.print(f'Accepted step: {step}. Decreased grad. norm from {norm0} to {norm1}')
                return x1
            else:
                step /= 2
                self.print(f'Decrease to {step}')
        self.exit_flag = True
        self.print('100 steps')
        # set_trace()
        return False


def cos_vec(x, y):
    return (x*y).sum() / ((x*x).sum()*(y*y).sum())**.5


def jac(fun, x, dx=1e-4):
    f0 = fun(x)
    dim_f = len(f0)
    dim_x = len(x)
    jac_f = np.zeros((dim_f, dim_x))
    for i in range(dim_x):
        x_ = np.copy(x)
        x_[i] += dx
        f_ = fun(x_)
        jac_f[i] = (f_ - f0)/dx
    return (jac_f + jac_f.T)/2
