import numpy as np


class CGLinear():
    def __init__(self, print_func=print):
        self.exit_flag = False
        self.print = print_func
        self.cost = {
            'objf': 0,
            'grads': 0,
            'steps': 0
        }

    def __call__(self, action, b, x0, iters=20, tol=1e-10):
        # x, status = self.run(action, b, x0, iters=iters, tol=tol)
        status = False
        if status is False:
            x, status = self.run(action, b, x0, iters=iters, tol=tol)
        if status is False:
            def action_(b):
                return action(action(b))

            b_ = action(b)
            x, status = self.run(action_, b_, x, iters=iters, tol=tol)
            # set_trace()
            pass
        return x

    def run(self, action, b, x0, iters=20, tol=1e-10):
        # x0 = np.float128(x0)
        # b = np.float128(b)
        # x = [x0]
        x = np.copy(x0)
        dims = np.prod(x0.shape)
        norm_b = np.abs(b).max()
        # self.print(f'CG - k:0, res: {np.abs(b - action(x)).max()}')
        abs_tol = np.maximum(tol * norm_b, 1e-14)
        k_ = 1
        for i in range(iters):
            rk = b - action(x)
            pk = rk
            for k in range(min(int(dims), 100)):
                act_pk = action(pk)
                step = (rk * rk).sum() / (pk * act_pk).sum()
                if np.isnan(step) or step < 0:
                    break
                x = x + step * pk
                rk_ = np.copy(rk)
                rk = rk_ - step * act_pk
                if np.abs(rk).max() < abs_tol:
                    self.print(
                        f'CG - Ended, k:{k_ + k}, res: {np.abs(rk).max()}')
                    self.cost['steps'] = k_ + k
                    return x, True
                beta = (rk * rk).sum() / (rk_ * rk_).sum()
                pk = rk + beta * pk
            k_ += k
        self.cost['steps'] = k_ + k
        self.print(f'CG - Ended, k:{k_ + k}, res: {np.abs(rk).max()}')
        # set_trace()
        return x, False
