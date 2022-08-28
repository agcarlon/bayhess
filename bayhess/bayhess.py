import numpy as np
from numpy.linalg import norm
from IPython.core.debugger import set_trace
from functools import partial
from .distributions import FrobeniusReg, Secant, DeterminantConstr, LogBarrier, \
    Wishart
from .cg import CGLinear
from .newton import Newton


class BayHess:
    def __init__(self, n_dim, strong_conv=0, smooth=1e10,
                 penal=1e-4, tol=1e-6, verbose=False, homotopy_steps=6,
                 yk_err_tol=0.7, reg_param=1e-4, finite=False,
                 pairs_to_use=None, ip_iters=10, bound_frac=.95,
                 datasize=None, sig_const=1e-3, cg_tol=1e-2,
                 homotopy_factor=2, check_curv_error=False, eps_yk=10,
                 prior='frobenius', relax_param=1.05, log=None):
        self.n_dim = n_dim
        self.relax_param = relax_param
        self.strong_conv = strong_conv / relax_param
        self.smooth = smooth * relax_param
        self.penal = penal
        self.tol = tol
        self.cg_tol = cg_tol
        self.homotopy_factor = homotopy_factor
        self.homotopy_steps = homotopy_steps
        self.yk_err_tol = yk_err_tol
        self.reg_param = reg_param
        self.ip_iters = ip_iters
        self.bound_frac = bound_frac
        if pairs_to_use is None:
            self.pairs_to_use = 10 * n_dim
        else:
            self.pairs_to_use = pairs_to_use
        self.hess = np.eye(n_dim) * (strong_conv + smooth) / 2
        self.inv_hess = np.eye(n_dim) * 2 / (strong_conv + smooth)
        # self.hess = np.eye(n_dim) * smooth
        # self.inv_hess = np.eye(n_dim) * 1 / smooth
        self.log = log
        self.verbose = verbose
        if log:
            logger = open(log, 'w')

            def logging(func, arg):
                logger.write(arg + '\n')
                func(arg)
        else:
            def logging(func, arg):
                func(arg)
        if verbose:
            self.print = lambda arg: logging(print, arg)
        else:
            no_print = lambda x: None
            self.print = lambda arg: logging(no_print, arg)
        self.update_hess_flag = False
        self.update_inv_flag = False
        self.finite = finite
        if finite and (datasize is None):
            raise Exception('If finite is True, set datasize argument.')
        self.datasize = datasize
        self.sig_const = sig_const
        self.check_curv_error = check_curv_error
        self.prior = prior
        self.xk = []
        self.yk = []
        self.sk = []
        self.pk_raw = []
        self.pk = []
        self.yk_all = []
        self.sk_all = []
        self.pk_all = []
        self.xk_all = []
        self.eps_yk = eps_yk

    def find_hess(self, x=None, full=False):
        if x is None:
            x = self.xk[-1]
        if not self.update_hess_flag:
            return self.hess
        self.update_hess_flag = False
        self.update_inv_flag = True
        # dists = norm(self.xk - x, axis=1)
        # weights = 1. / (dists + 1e-20)
        # weights = weights / np.sum(weights)
        # self.pk = np.multiply(np.transpose(self.pk_raw), weights).T
        # self.pk = np.asarray(self.pk_raw)/np.sum(self.pk_raw)
        self.pk = np.asarray(self.pk_raw)
        factor = 0
        for s, y, p in zip(self.sk, self.yk, self.pk):
            factor += norm(p * (self.hess @ s - y)) * norm(s)
        lkl = Secant(self.sk, self.yk, self.pk / factor)
        # if self.prior is 'Wishart':
        if self.prior == 'wishart':
            extra_dof = 2.
            dof = self.n_dim + extra_dof
            prior_1 = Wishart(self.hess/(self.n_dim - dof - 1), dof)
        elif self.prior == 'frobenius':
            prior_1 = FrobeniusReg(self.hess, np.eye(self.n_dim) * self.reg_param)

        prior_2 = LogBarrier(self.strong_conv, self.smooth,
                             self.penal * self.homotopy_factor ** self.homotopy_steps)
        post = prior_1 * prior_2 * lkl
        self.post = post
        opt = Newton(print_func=self.print, ls='backtrack_armijo')
        # opt = Newton(print_func=self.print, ls='wolfe_armijo')
        opt.bay = self
        inv_hess_action = partial(self.inv_action, post.hess_log_pdf_action,
                                  print_func=self.print, tol=self.cg_tol)
        hess = self.hess.copy()
        self.print(f'Starting Bay-Hess algorithm: {len(self.sk)} curv. pairs')
        for h in range(self.homotopy_steps):
            prior_2.penal = self.penal * self.homotopy_factor ** (
                    self.homotopy_steps - h - 1)
            tol = self.tol * self.homotopy_factor ** (
                    self.homotopy_steps - h - 1)
            self.print(
                f'Starting homotopy:{h}, penal:{prior_2.penal}, tol:{tol}')
            hess = opt.run_matrix(post.log_pdf,
                                  post.grad_log_pdf,
                                  inv_hess_action,
                                  hess,
                                  tol=tol,
                                  iters=1000)
        self.hess = (hess + hess.T) / 2
        return self.hess

    def inv_action(self, action, x, b, print_func=print, tol=1e-5):
        opt = CGLinear(print_func=print_func)

        def act_x(x_):
            return action(x, x_)

        x_ = opt(act_x, b, np.zeros(b.shape), tol=tol)
        return x_

    def update_curv_pairs(self, sk, yk, xk):
        if self.finite:
            sigma = np.var(yk, axis=0, ddof=1).sum() / len(yk) \
                    * (self.datasize - len(yk)) / (self.datasize - 1)
        else:
            sigma = np.var(yk, axis=0, ddof=1).sum() / len(yk)
        if sigma < self.yk_err_tol * norm(yk) ** 2:
            sigma += self.sig_const * np.max(sigma) + 1e-8
            pk_k = 1. / sigma
            if np.isinf(pk_k):
                print(1)
            self.sk.append(sk)
            self.yk.append(yk.mean(axis=0))
            self.pk_raw.append(pk_k)
            self.xk.append(xk - sk / 2)
            self.update_hess_flag = True
            self.sk = self.sk[-self.pairs_to_use:]
            self.yk = self.yk[-self.pairs_to_use:]
            self.xk = self.xk[-self.pairs_to_use:]
            self.pk_raw = self.pk_raw[-self.pairs_to_use:]

    def update_curv_pairs_mice(self, df, xk):
        for delta in df.deltas[1:]:
            if self.finite:
                sigma = delta.m2_del / (delta.m - 1) / delta.m \
                        * (self.datasize - delta.m) / (self.datasize - 1)
            else:
                sigma = delta.m2_del / (delta.m - 1) / delta.m
            if np.sum(sigma) < self.yk_err_tol * norm(delta.f_delta_av) ** 2:
                if not hasattr(delta, 'curv_pair'):
                    self.sk_all.append(delta.x_l - delta.x_l1)
                    self.yk_all.append(None)
                    self.pk_all.append(None)
                    self.xk_all.append((delta.x_l + delta.x_l1) / 2)
                    delta.curv_pair = len(self.sk_all) - 1
                if self.check_curv_error:
                    self.check_mice_curv_error(delta, df)
                self.yk_all[delta.curv_pair] = delta.f_delta_av
                sigma += self.sig_const * np.max(np.abs(sigma)) + 1e-8
                pk_k = 1. / sigma
                self.pk_all[delta.curv_pair] = pk_k
                self.update_hess_flag = True
        self.sk = self.sk_all[-self.pairs_to_use:]
        self.yk = self.yk_all[-self.pairs_to_use:]
        self.xk = self.xk_all[-self.pairs_to_use:]
        self.pk_raw = self.pk_all[-self.pairs_to_use:]

    def check_mice_curv_error(self, delta, df):
        df.print(
            (f'Checking curvature pair {delta.curv_pair} statistical error'))
        norm_yk = df.norm(delta.f_delta_av)
        norms = df.norm(delta.f_deltas, axis=1)
        norms = np.append(norms, norm_yk)
        norms = np.sort(norms)
        norm_yk = norms[int(len(norms) * df.re_percentile)]
        std_norm_yk = np.sqrt(delta.v_l)
        C_I2 = (std_norm_yk / (self.eps_yk * norm_yk)) ** 2
        if df.finite:
            m_need = np.ceil(C_I2 * df.data_size / (C_I2 + df.data_size))
        else:
            m_need = np.ceil(C_I2)
        m_now = delta.m
        while m_now < m_need:
            m_to_samp = np.ceil(np.min([m_need - m_now, m_now])).astype('int')
            if df._check_max_cost(
                    extra_eval=m_to_samp * delta.c) or df.terminate:
                return
            self.print((f'delta sample size {m_now} is not large enough to '
                        f'evaluate curvature pair {delta.curv_pair}, '
                        f'increasing to {m_now + m_to_samp}'))
            delta.update_delta(df, m_now + m_to_samp)
            m_now = delta.m
            norm_yk = df.norm(delta.f_delta_av)
            norms = df.norm(delta.f_deltas, axis=1)
            norms = np.append(norms, norm_yk)
            norms = np.sort(norms)
            norm_yk = norms[int(len(norms) * df.re_percentile)]
            std_norm_yk = np.sqrt(delta.v_l)
            C_I2 = (std_norm_yk / (self.eps_yk * norm_yk)) ** 2
            if df.finite:
                m_need = np.ceil(C_I2 * df.data_size /
                                 (C_I2 + df.data_size))
            else:
                m_need = np.ceil(C_I2)
        df.print(f'delta sample size {m_now} is large enough to evaluate '
                 f'curvature pair {delta.curv_pair}, sample size needed is '
                 f'{m_need}')
        return

    def eval_inverse_hess(self):
        if not self.update_inv_flag:
            return self.inv_hess
        self.update_inv_flag = False
        tol = 1e-4
        inv = self.inv_hess.copy()
        res = norm(inv @ self.hess - np.eye(self.n_dim))
        k = 0
        ress = [res]
        while res > tol and k < 100:
            k += 1
            inv = 2 * inv - inv @ self.hess @ inv
            res_ = norm(inv @ self.hess - np.eye(self.n_dim))
            if np.isnan(inv).any() or np.isnan(res):
                set_trace()
            if res_ >= res:
                inv = np.eye(self.n_dim) * 2 / (self.strong_conv + self.smooth)
                res = norm(inv @ self.hess - np.eye(self.n_dim))
                ress.append(res)
            else:
                res = res_.copy()
                ress.append(res)
        self.inv_hess = inv
        return inv
