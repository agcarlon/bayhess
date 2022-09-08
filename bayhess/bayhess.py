import numpy as np
from numpy.linalg import norm
from functools import partial
from .distributions import FrobeniusReg, Secant, SecantInverse, LogBarrier, Wishart
from .cg import CGLinear
from .newton import Newton


class BayHess:
    """
    Class for Bayesian Hessian estimators

    Parameters
    ----------
    n_dim : int
        Dimensionality of optimization variables
    stron_conv : float, default=0.
        Strong convexity parameter of objective function
    smooth : float, default=1e10
        Smoothness of objective function
    penal : float, default=1e-4
        Penalization parameter for logarithmic barrier constraint on the
        Hessian eigenvalues
    tol : float, default=1e-6
        Tolerance on the norm of the gradient (of the log posterior). If the
        norm of the gradient is less than tol, the Hessian aproximation found
        is returned
    verbose : bool, default=False
        Verbosity
    homotopy_steps : int, default=6
        Number of hotomopy steps taken to find the Hessian, i.e.,
        number of times the Newton-CG method is used with decreasing
        logarithmic barrier parameters
    yk_err_tol : float, default=0.7
         Criterion to use curvature pair. If the variance of the gradient
         differences summed component-wise is larger than yk_err_tol times
         the norm of mean gradient differences squared, the curvature pair is
         ignored
    reg_param : float, default=1e-4
        Parameter for the Frobenius norm regularization prior term
    finite : bool, default=False
        If True, minimizing a finite sum instead of an expectation
    pairs_to_use : int, default=None
        Number of curvature pairs to keep in memory and use to compute the
        Hessian approximation. If None, 10 times the dimensionality is used
    data_size : int, default=None
        If the problem is one of finite sum optimization, data_size defines
        the number of summands
    sigma_constant : float, default=1e-3
        Constant to be added to the diagonals of the covariance matrix to
        make it non-singular
    cg_tol : float, default=1e-2
        Tolerance on the norm of the residue to stop the conjugate gradient
        method used to find the Newton direction in the Newton-CG method.
    homotopy_factor : int, default=2
        Factor to which the logarithmic barrier parameter and Newton-CG
        tolerance are decreased each homotopy step
    check_curv_error : bool, default=False
        If True, forces relative statistical error of each curvature error to
        be less than 'eps_yk'
    eps_yk : float, default=10.
        Tolerance on the relative statistical error of each curvature pair if
        check_curv_error is True
    prior : {'frobenius', 'wishart'}, default='frobenius'
        Prior distribution of the Hessian.
    relax_param : float, default=1.05
        Parameter to relax the logarithmic barriers of the eigenvalues of the
        Hessian. If set to 1.0, the logarithmic barrier is exactly on top of
        the eigenvalues extremes, i.e., the prior distribution has probability
        zero of observing a Hessian with the extreme eigenvalues. Should be >1.
    log : str, default=None
        File to write log of outputs
    """

    def __init__(self, n_dim, strong_conv=0., smooth=1e10,
                 penal=1e-2, tol=1e-6, verbose=False, homotopy_steps=6,
                 yk_err_tol=0.7, reg_param=1e-2, finite=False,
                 pairs_to_use=None,
                 data_size=None, sigma_constant=1e-3, cg_tol=1e-2,
                 homotopy_factor=2, check_curv_error=False, eps_yk=10.,
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
        if finite and (data_size is None):
            raise Exception('If finite is True, set data_size argument.')
        self.data_size = data_size
        self.sigma_constant = sigma_constant
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

    def find_hess(self):
        """
        Finds Hessian approximation.

        Returns
        -------
        Hessian estimate: array_like
            Hessian approximation
        """
        if not self.update_hess_flag:
            return self.hess
        self.update_hess_flag = False
        self.update_inv_flag = True
        self.pk = np.asarray(self.pk_raw)
        factor = 0
        for s, y, p in zip(self.sk, self.yk, self.pk):
            factor += norm(p * (self.hess @ s - y)) * norm(s)
        lkl = Secant(self.sk, self.yk, self.pk / factor)
        if self.prior == 'wishart':
            extra_dof = 2.
            dof = self.n_dim + extra_dof
            prior_1 = Wishart(self.hess / (self.n_dim - dof - 1), dof)
        elif self.prior == 'frobenius':
            prior_1 = FrobeniusReg(self.hess,
                                   np.eye(self.n_dim) * self.reg_param)

        prior_2 = LogBarrier(self.strong_conv, self.smooth,
                             self.penal * self.homotopy_factor ** self.homotopy_steps)
        post = prior_1 * prior_2 * lkl
        opt = Newton(print_func=self.print, ls='backtrack_armijo')
        # opt = Newton(print_func=self.print, ls='wolfe_armijo')
        opt.bay = self
        inv_hess_action = partial(self._inv_action, post.hess_log_pdf_action,
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

    def _inv_action(self, action, x, b, print_func=print, tol=1e-5):
        opt = CGLinear(print_func=print_func)

        def act_x(x_):
            return action(x, x_)

        x_ = opt(act_x, b, np.zeros(b.shape), tol=tol)
        return x_

    def update_curv_pairs(self, sk, yk, xk):
        """
        Update the curvature pairs

        Parameters
        ----------
        sk : array_like
            Difference of points on space where the gradients are computed
        yk : 2-d array_like
            Array with sample of gradient differences. Shape should be
            (sample_size, dimensions).
        xk : array_like
            Current iterate
        """
        if self.finite:
            sigma = np.var(yk, axis=0, ddof=1).sum() / len(yk) \
                    * (self.data_size - len(yk)) / (self.data_size - 1)
        else:
            sigma = np.var(yk, axis=0, ddof=1).sum() / len(yk)
        if sigma < self.yk_err_tol * norm(yk) ** 2:
            sigma += self.sigma_constant * np.max(sigma) + 1e-8
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
        """
        Update the curvature pairs from MICE object

        Parameters
        ----------
        df : MICE object
            Instance of MICE used to compute the gradients. Curvature pairs
            are extracted from gradient differences at MICE hierarchy.
        xk : array_like
            Current iterate
        """
        for delta in df.deltas[1:]:
            if self.finite:
                sigma = delta.m2_del / (delta.m - 1) / delta.m \
                        * (self.data_size - delta.m) / (self.data_size - 1)
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
                    self._check_mice_curv_error(delta, df)
                self.yk_all[delta.curv_pair] = delta.f_delta_av
                sigma += self.sigma_constant * np.max(np.abs(sigma)) + 1e-8
                pk_k = 1. / sigma
                self.pk_all[delta.curv_pair] = pk_k
                self.update_hess_flag = True
        self.sk = self.sk_all[-self.pairs_to_use:]
        self.yk = self.yk_all[-self.pairs_to_use:]
        self.xk = self.xk_all[-self.pairs_to_use:]
        self.pk_raw = self.pk_all[-self.pairs_to_use:]

    def _check_mice_curv_error(self, delta, df):
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
        """
        Function used to compute the inverse Hessian from the Hessian
        approximation. It uses Newton--Raphson starting from current Hessian
        inverse approximation.
        """
        if not self.update_inv_flag:
            return self.inv_hess
        self.update_inv_flag = False
        tol = 1e-6
        inv = self.inv_hess.copy()
        res = norm(inv @ self.hess - np.eye(self.n_dim))
        k = 0
        ress = [res]
        while res > tol and k < 100:
            k += 1
            inv = 2 * inv - inv @ self.hess @ inv
            res_ = norm(inv @ self.hess - np.eye(self.n_dim))
            if np.isnan(inv).any() or np.isnan(res):
                raise ValueError("Inverse computation did not converge.")
            if res_ >= res:
                inv = np.eye(self.n_dim) * 2 / (self.strong_conv + self.smooth)
                res = norm(inv @ self.hess - np.eye(self.n_dim))
                ress.append(res)
            else:
                res = res_.copy()
                ress.append(res)
        self.inv_hess = inv
        return inv
