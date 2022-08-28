import numpy as np
from numpy.linalg import det, inv
from scipy.linalg import eigvalsh


class Distribution:
    def __init__(self, log_pdf, grad_log_pdf, hess_log_pdf_action,
                 hess_log_pdf, secant_flag=False):
        self.log_pdf = log_pdf
        self.grad_log_pdf = grad_log_pdf
        self.hess_log_pdf_action = hess_log_pdf_action
        self.hess_log_pdf = hess_log_pdf
        self.secant_flag = secant_flag
        self.construct = [self]
        self.likelihood = []

    def __mul__(self, other):
        def log_pdf(hess):
            return self.log_pdf(hess) + other.log_pdf(hess)

        def grad_log_pdf(hess):
            return self.grad_log_pdf(hess) + other.grad_log_pdf(hess)

        def hess_log_pdf_action(hess, direc):
            return self.hess_log_pdf_action(hess, direc) + \
                   other.hess_log_pdf_action(hess, direc)

        def hess_log_pdf(hess):
            return self.hess_log_pdf(hess) + other.hess_log_pdf(hess)

        out = Distribution(log_pdf, grad_log_pdf, hess_log_pdf_action,
                           hess_log_pdf)
        out.construct = [*self.construct, *other.construct]
        out.likelihood = [obj for obj in out.construct if obj.secant_flag]
        return out

    def update_curv_pairs(self, sk, yk):
        for lkl in self.likelihood:
            lkl.update_lkl_curv_pairs(sk, yk)
        else:
            raise Exception('There is no Secant object in Distribution.')


class FrobeniusReg(Distribution):
    '''
    Class of FrobeniusReg distributions
    '''

    def __init__(self, mean, weights):
        self.mean = mean
        self.weights = weights
        self.ident = None
        self.n_dim = len(mean)
        super().__init__(self.log_pdf, self.grad_log_pdf,
                         self.hess_log_pdf_action, self.hess_log_pdf)

    def log_pdf(self, hess):
        return np.sum((self.weights @ (hess - self.mean) @ self.weights) ** 2)

    def grad_log_pdf(self, hess):
        aux = 2 * self.weights @ self.weights @ (
                    hess - self.mean) @ self.weights @ self.weights
        return (aux + aux.T) / 2

    def hess_log_pdf_action(self, hess, direc):
        aux = 2 * self.weights @ self.weights @ direc @ self.weights @ self.weights
        return (aux + aux.T) / 2

    # Not working, for some reason, so, using directional deriv.
    def hess_log_pdf(self, hess):
        if self.ident is None:
            self.ident = np.einsum('ik,jl', np.eye(self.n_dim),
                                   np.eye(self.n_dim))
            self.ident = (
                                 self.ident + np.transpose(self.ident,
                                                           (1, 0, 2, 3))) / 2
            self.ident = (
                                 self.ident + np.transpose(self.ident,
                                                           (0, 1, 3, 2))) / 2
        # aux = np.einsum('ni, ijkl, lm -> njkm', self.weights@self.weights, self.ident, self.weights@self.weights) /2
        #
        # aux = (aux + np.transpose(aux, (1, 0, 2, 3))) / 2
        # aux = (aux + np.transpose(aux, (0, 1, 3, 2))) / 2

        hh = np.zeros(self.ident.shape)
        for i, id in enumerate(self.ident):
            for j, id_ in enumerate(id):
                hh[i, j] = self.hess_log_pdf_action(hess, id_)

        return hh


class Wishart(Distribution):
    '''
    Class of Wishart distributions
    '''

    def __init__(self, scale, dof):
        self.scale_inv = inv(scale)  # V^-1
        self.dof = dof  # n
        self.ident = None
        self.n_dim = len(scale)  # p
        if dof <= self.n_dim - 1:
            raise ValueError(
                "Degrees of freedom must be dof > p-1 for problem "
                "of dimension p.")
        super().__init__(self.log_pdf, self.grad_log_pdf,
                         self.hess_log_pdf_action, self.hess_log_pdf)

    def log_pdf(self, hess):
        return (-self.dof + self.n_dim + 1) / 2 * np.log(det(hess)) \
               + np.sum(self.scale_inv * hess) / 2

    def grad_log_pdf(self, hess):
        aux = (-self.dof + self.n_dim + 1) / 2 * inv(hess) + self.scale_inv / 2
        return (aux + aux.T) / 2

    def hess_log_pdf_action(self, hess, direc):
        aux = inv(hess)
        aux_ = -(1 + self.n_dim - self.dof) / 2 * (aux @ direc @ aux)
        return (aux_ + aux_.T) / 2

    # Not working, for some reason, so, using directional deriv.
    def hess_log_pdf(self, hess):
        if self.ident is None:
            self.ident = np.einsum('ik,jl', np.eye(self.n_dim),
                                   np.eye(self.n_dim))
            self.ident = (
                                 self.ident + np.transpose(self.ident,
                                                           (1, 0, 2, 3))) / 2
            self.ident = (
                                 self.ident + np.transpose(self.ident,
                                                           (0, 1, 3, 2))) / 2

        hh = np.zeros(self.ident.shape)
        for i, id in enumerate(self.ident):
            for j, id_ in enumerate(id):
                hh[i, j] = self.hess_log_pdf_action(hess, id_)

        # aux = inv(hess)
        # return -(1 + self.n_dim - self.dof)/2 * np.einsum('ij,kl->ijkl', aux, aux)
        return hh


class Secant(Distribution):
    def __init__(self, sk, yk, pk):
        self.sk = sk
        self.yk = yk
        self.pk = pk
        self.n_dim = len(self.sk[0])
        super().__init__(self.log_pdf, self.grad_log_pdf,
                         self.hess_log_pdf_action, self.hess_log_pdf,
                         secant_flag=True)
        self.likelihood = [self]

    def log_pdf(self, hess):
        log_pdf = 0
        for s, y, p in zip(self.sk, self.yk, self.pk):
            res = hess @ s - y
            log_pdf += res @ (p * res)
        log_pdf *= .5
        return log_pdf

    def grad_log_pdf(self, hess):
        grad_log_pdf = np.zeros((self.n_dim, self.n_dim))
        for s, y, p in zip(self.sk, self.yk, self.pk):
            res = hess @ s - y
            grad_log_pdf += np.outer((p * res), s)
        return (grad_log_pdf + grad_log_pdf.T) / 2

    def hess_log_pdf_action(self, hess, direc):
        action = np.zeros((self.n_dim, self.n_dim))
        for s, y, p in zip(self.sk, self.yk, self.pk):
            act_lkl = np.outer(p * (direc @ s), s)
            action += (act_lkl + act_lkl.T) / 2
        return action

    def hess_log_pdf(self, hess):
        hess_log_pdf = 0
        for s, y, p in zip(self.sk, self.yk, self.pk):
            hess_log_pdf += np.einsum('in,l,j->ilnj', np.diag(p), s, s)
        hess_log_pdf = (
                               hess_log_pdf + np.transpose(hess_log_pdf,
                                                           (1, 0, 2, 3))) / 2
        hess_log_pdf = (
                               hess_log_pdf + np.transpose(hess_log_pdf,
                                                           (0, 1, 3, 2))) / 2
        return hess_log_pdf

    def variance_norm(self, hess):
        hess_log_pdf = 0
        for s, y, p in zip(self.sk, self.yk, self.pk):
            hess_log_pdf += np.einsum('in,l,j->ilnj', np.diag(p), s, s)

    def update_lkl_curv_pairs(self, sk, yk):
        pass


class SecantInverse(Distribution):
    def __init__(self, sk, yk, pk):
        self.sk = sk
        self.yk = yk
        self.pk = pk
        self.n_dim = len(self.sk[0])
        super().__init__(self.log_pdf, self.grad_log_pdf,
                         self.hess_log_pdf_action, self.hess_log_pdf,
                         secant_flag=True)
        self.likelihood = [self]

    def log_pdf(self, hess_inv):
        log_pdf = 0
        # hess_inv_det = det(hess_inv)
        # hess_inv_det = 1
        for s, y, p in zip(self.sk, self.yk, self.pk):
            p_ = p
            res = hess_inv @ y - s
            log_pdf += res @ (p_ * res)
        log_pdf *= .5
        return log_pdf

    def grad_log_pdf(self, hess_inv):
        hess_inv_det = det(hess_inv)
        grad_log_pdf = np.zeros((self.n_dim, self.n_dim))
        for s, y, p in zip(self.sk, self.yk, self.pk):
            p_ = p
            res = hess_inv @ y - s
            grad_log_pdf += np.outer((p_ * res), y)
        return (grad_log_pdf + grad_log_pdf.T) / 2

    def hess_log_pdf_action(self, hess, direc):
        action = np.zeros((self.n_dim, self.n_dim))
        for s, y, p in zip(self.sk, self.yk, self.pk):
            p_ = p
            act_lkl = np.outer(p_ * (direc @ y), y)
            action += (act_lkl + act_lkl.T) / 2
        return action

    def hess_log_pdf(self, hess):
        hess_log_pdf = 0
        for s, y, p in zip(self.sk, self.yk, self.pk):
            p_ = p
            hess_log_pdf += np.einsum('in,l,j->ilnj', np.diag(p_), y, y)
        hess_log_pdf = (
                               hess_log_pdf + np.transpose(hess_log_pdf,
                                                           (1, 0, 2, 3))) / 2
        hess_log_pdf = (
                               hess_log_pdf + np.transpose(hess_log_pdf,
                                                           (0, 1, 3, 2))) / 2
        return hess_log_pdf

    def variance_norm(self, hess):
        hess_log_pdf = 0
        for s, y, p in zip(self.sk, self.yk, self.pk):
            p_ = p
            hess_log_pdf += np.einsum('in,l,j->ilnj', np.diag(p_), y, y)

    def update_lkl_curv_pairs(self, sk, yk):
        pass


class LogBarrier(Distribution):
    def __init__(self, lower, upper, penal):
        self.lower = lower
        self.upper = upper
        self.penal = penal
        self.num_const = 2
        super().__init__(self.log_pdf, self.grad_log_pdf,
                         self.hess_log_pdf_action, self.hess_log_pdf)

    def log_pdf(self, hess):
        n_dim = len(hess)
        # min_eig = eigvalsh(hess, subset_by_index=[0, 1])
        # max_eig = eigvalsh(hess, subset_by_index=[n_dim-2, n_dim-1])
        eigs = eigvalsh(hess)
        if eigs.min() < self.lower or eigs.max() > self.upper:
            return np.inf
        g = -self.penal * \
            np.linalg.slogdet(hess - self.lower * np.eye(n_dim))[1]
        g += -self.penal * \
             np.linalg.slogdet(self.upper * np.eye(n_dim) - hess)[1]
        if np.isnan(g):
            g = np.inf
        return g

    def grad_log_pdf(self, hess):
        n_dim = len(hess)
        diff_g = - self.penal * \
                 np.linalg.inv(hess - self.lower * np.eye(n_dim))
        diff_g += self.penal * \
                  np.linalg.inv(self.upper * np.eye(n_dim) - hess)
        return (diff_g + diff_g.T) / 2.

    def hess_log_pdf_action(self, hess, direc):
        n_dim = len(hess)
        shift_l = np.linalg.inv(hess - self.lower * np.eye(n_dim))
        shift_u = np.linalg.inv(self.upper * np.eye(n_dim) - hess)
        # vec = self.matrix(vec)
        action = self.penal * (shift_l @ direc @ shift_l)
        action += self.penal * (shift_u @ direc @ shift_u)
        return (action + action.T) / 2.

    def hess_log_pdf(self, hess):
        n_dim = len(hess)
        aux = np.linalg.inv(hess - self.lower * np.eye(n_dim))
        aux2 = np.linalg.inv(self.upper * np.eye(n_dim) - hess)
        hess_g = self.penal * np.einsum('in,lj->nlij', aux, aux)
        hess_g += self.penal * np.einsum('in,lj->nlij', aux2, aux2)
        hess_g = (
                         hess_g + np.transpose(hess_g, (1, 0, 2, 3))) / 2
        hess_g = (
                         hess_g + np.transpose(hess_g, (0, 1, 3, 2))) / 2
        return hess_g


class DeterminantConstr():
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.num_const = 2

    def __call__(self, hess):
        n_dim = len(hess)
        g = np.zeros(2)
        g[0] = np.linalg.det(hess - self.lower * np.eye(n_dim))
        g[1] = np.linalg.det(self.upper * np.eye(n_dim) - hess)
        return g

    def grad(self, hess):
        n_dim = len(hess)
        diff_g = [np.linalg.inv(hess - self.lower * np.eye(n_dim))
                  * np.linalg.det(hess - self.lower * np.eye(n_dim))]
        diff_g.append(- np.linalg.inv(self.upper * np.eye(n_dim) - hess)
                      * np.linalg.det(self.upper * np.eye(n_dim) - hess))
        return np.array(diff_g)

    def hess_action(self, hess, direc):
        n_dim = len(hess)
        inv_l = np.linalg.inv(hess - self.lower * np.eye(n_dim))
        det_l = np.linalg.det(hess - self.lower * np.eye(n_dim))
        inv_u = np.linalg.inv(self.upper * np.eye(n_dim) - hess)
        det_u = np.linalg.det(self.upper * np.eye(n_dim) - hess)

        action = [-(inv_l @ direc @ inv_l) * det_l
                  + ((inv_l * direc).sum() * det_l) * inv_l]
        action.append(-(inv_u @ direc @ inv_u) * det_u
                      + ((inv_u * direc).sum() * det_u) * inv_u)
        return np.array(action)

    def hess(self, hess):
        raise Exception('The Hessian of the log pdf of the Determinant '
                        'constraint class has not been implemented yet')


def test_derivatives(objf, grad, n_dim, n_out=1):
    print('Testing first order derivatives')
    np.random.seed(1)
    A = np.random.rand(n_dim, n_dim)
    A = A @ A.T
    jac = grad(A)
    # set_trace()
    dxs = [1e-3, 1e-4, 1e-5, 1e-6]
    errors = []
    for dx in dxs:
        if n_out > 1:
            jac_df = np.zeros((n_out, n_dim, n_dim))
            for i in range(n_dim):
                for j in range(n_dim):
                    A_ = A.copy()
                    A_[i, j] += dx / 2
                    A_[j, i] += dx / 2
                    jac_df[:, i, j] = (objf(A_) - objf(A)) / dx
        else:
            jac_df = np.zeros((n_dim, n_dim))
            for i in range(n_dim):
                for j in range(n_dim):
                    A_ = A.copy()
                    A_[i, j] += dx / 2
                    A_[j, i] += dx / 2
                    jac_df[i, j] = (objf(A_) - objf(A)) / dx
        errors.append(np.linalg.norm(jac_df - jac))
        print(errors[-1])


def test_hess_act(grad, hess_act, n_dim):
    print('Testing Hessian action')
    np.random.seed(1)
    A = np.random.rand(n_dim, n_dim)
    A = A @ A.T
    V = np.random.rand(n_dim, n_dim)
    V = V @ V.T
    act = hess_act(A, V)
    dxs = [1e-3, 1e-4, 1e-5, 1e-6]
    errors = []
    for dx in dxs:
        A_ = A + dx * V
        act_df = (grad(A_) - grad(A)) / dx
        errors.append(np.linalg.norm(act_df - act))
        print(errors[-1])


def test_hess(grad, hess, n_dim):
    print('Testing Hessian')
    np.random.seed(1)
    A = np.random.rand(n_dim, n_dim)
    A = A @ A.T
    hess = hess(A)
    dxs = [1e-3, 1e-4, 1e-5, 1e-6]
    errors = []
    for dx in dxs:
        hess_df = np.zeros((n_dim, n_dim, n_dim, n_dim))
        for i in range(n_dim):
            for j in range(n_dim):
                A_ = A.copy()
                A_[i, j] += dx / 2
                A_[j, i] += dx / 2
                hess_df[i, j] = (grad(A_) - grad(A)) / dx
        # hess_df = (hess_df + np.transpose(hess_df, (1, 0, 2, 3))) / 2
        # hess_df = (hess_df + np.transpose(hess_df, (0, 1, 3, 2))) / 2
        errors.append(np.linalg.norm(hess_df - hess))
        print(errors[-1])


def test_hess_(hess, hess_act, n_dim):
    print('Testing Hessian from Hessian action')
    np.random.seed(1)
    A = np.random.rand(n_dim, n_dim)
    A = A @ A.T
    V = np.random.rand(n_dim, n_dim)
    V = V @ V.T
    hess = hess(A)
    act = np.einsum('ijkl,kl', hess, V)
    act_ = hess_act(A, V)
    error = np.linalg.norm(act - act_)
    print(error)


def test_regularizer():
    print('Testing FrobeniusReg class derivatives')
    np.random.seed(0)
    n_dim = 3
    mean = np.random.randn(n_dim, n_dim)
    mean = mean @ mean.T
    weights = np.random.randn(n_dim, n_dim)
    weights = weights @ weights.T
    normal = FrobeniusReg(mean, weights)
    test_derivatives(normal.log_pdf, normal.grad_log_pdf, n_dim)
    test_hess_act(normal.grad_log_pdf, normal.hess_log_pdf_action, n_dim)
    test_hess(normal.grad_log_pdf, normal.hess_log_pdf, n_dim)
    test_hess_(normal.hess_log_pdf, normal.hess_log_pdf_action, n_dim)


def test_wishart():
    print('Testing Wishart class derivatives')
    np.random.seed(0)
    n_dim = 3
    mean = np.random.randn(n_dim, n_dim)
    mean = mean @ mean.T
    normal = Wishart(mean / n_dim, n_dim)
    test_derivatives(normal.log_pdf, normal.grad_log_pdf, n_dim)
    test_hess_act(normal.grad_log_pdf, normal.hess_log_pdf_action, n_dim)
    test_hess(normal.grad_log_pdf, normal.hess_log_pdf, n_dim)
    test_hess_(normal.hess_log_pdf, normal.hess_log_pdf_action, n_dim)


def test_secant():
    print('Testing Secant class derivatives')
    np.random.seed(0)
    n_dim = 3
    n_points = 6
    mean = np.random.randn(n_dim)
    hess = np.random.randn(n_dim, n_dim)
    hess = hess @ hess.T
    gr = lambda x: hess @ (x - mean)

    xk = np.random.randn(n_points, n_dim)
    grs = np.array([gr(x) for x in xk])
    yk = grs[1:] - grs[:-1]
    sk = xk[1:] - xk[:-1]
    pks = np.random.rand(n_points - 1, n_dim)

    lkl = Secant(sk, yk, pks)
    test_derivatives(lkl.log_pdf, lkl.grad_log_pdf, n_dim)
    test_hess_act(lkl.grad_log_pdf, lkl.hess_log_pdf_action, n_dim)
    test_hess(lkl.grad_log_pdf, lkl.hess_log_pdf, n_dim)
    test_hess_(lkl.hess_log_pdf, lkl.hess_log_pdf_action, n_dim)


def test_secant_inv():
    print('Testing SecantInverse class derivatives')
    np.random.seed(0)
    n_dim = 3
    n_points = 6
    mean = np.random.randn(n_dim)
    hess = np.random.randn(n_dim, n_dim)
    hess = hess @ hess.T
    gr = lambda x: hess @ (x - mean)

    xk = np.random.randn(n_points, n_dim)
    grs = np.array([gr(x) for x in xk])
    yk = grs[1:] - grs[:-1]
    sk = xk[1:] - xk[:-1]
    pks = np.random.rand(n_points - 1, n_dim)

    lkl = Secant(sk, yk, pks)
    test_derivatives(lkl.log_pdf, lkl.grad_log_pdf, n_dim)
    test_hess_act(lkl.grad_log_pdf, lkl.hess_log_pdf_action, n_dim)
    test_hess(lkl.grad_log_pdf, lkl.hess_log_pdf, n_dim)
    test_hess_(lkl.hess_log_pdf, lkl.hess_log_pdf_action, n_dim)


def test_log_barrier():
    print('Testing LogBarrier class derivatives')
    np.random.seed(0)
    n_dim = 3
    lower = 1e-4
    upper = 1.
    penal = 1e-2
    logbarr = LogBarrier(lower=lower, upper=upper, penal=penal)
    test_derivatives(logbarr.log_pdf, logbarr.grad_log_pdf, n_dim)
    test_hess_act(logbarr.grad_log_pdf, logbarr.hess_log_pdf_action, n_dim)
    test_hess(logbarr.grad_log_pdf, logbarr.hess_log_pdf, n_dim)
    test_hess_(logbarr.hess_log_pdf, logbarr.hess_log_pdf_action, n_dim)


def test_det_constr():
    print('Testing DeterminantConstr class derivatives')
    np.random.seed(0)
    n_dim = 3
    lower = 1e-4
    upper = 1.
    constr = DeterminantConstr(lower=lower, upper=upper)
    test_derivatives(constr, constr.grad, n_dim, n_out=2)
    test_hess_act(constr.grad, constr.hess_action, n_dim)


if __name__ == '__main__':
    test_regularizer()
    test_wishart()
    test_secant()
    test_secant_inv()
    test_log_barrier()
    test_det_constr()
