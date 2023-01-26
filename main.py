from typing import Callable, TypeAlias, TypeVar
import numpy as np
from scipy.integrate import quad
from functools import partial


def lagrangian_polynomial(p: int, j: int, nodal_points: np.ndarray) -> Callable[[float], float]:

    def sub_f(i: int, j: int):
        return lambda x: (x-nodal_points[i-1])/(nodal_points[j] - nodal_points[i])

    def poly(x: float):
        prod = 1

        for i in range(1, p-2):
            if i == j:
                continue

            prod *= sub_f(i, j)(x)

        return prod

    return poly


def compute_initial_nodal_points(s: int) -> list[int]:
    p = 2*s+1
    h = 1/(p-1)

    return np.fromfunction(lambda _, j: j*h, (1, p), dtype=float)[0]


def compute_b_matrix(p: int, nodal_points):

    b_matrix = np.zeros(shape=(p, p))

    def integrand_of_poly(poly):
        return lambda x: quad(poly, 0, x)[0]

    a = 1/nodal_points[p-1]
    b = nodal_points[p-1]

    for i in range(p):
        for j in range(p):
            integrand = integrand_of_poly(
                    lagrangian_polynomial(p, j, nodal_points)
            )

            c = quad(integrand, 0, nodal_points[i-1])[0]
            d = nodal_points[i]
            e = quad(integrand, 0, nodal_points[p-1])[0]

            b_matrix[i][j] = a * (b*c - d*e)

    return b_matrix


def sigma(b_matrix, phi, k: int, s: int):
    """computes sigma at ks+1 """

    s1 = 0
    for t in range(1, k-1):
        _s = 0
        for j in range(1, 2*s-1):
            a = b_matrix[s][j]
            b = phi[(t-1)*s+j]
            _s += a*b

        s1 += t * _s

    s2 = 0
    for j in range(1, 2*s-1):
        a = b_matrix[s][j-1]
        b = phi[(k-1)*s+j-1]
        s2 += a*b

    s2 = k * s2

    s3 = 0
    for t in range(1, k-1):
        _s = 0
        for j in range(1, 2*s-1):
            a = b_matrix[s][j]
            b = phi[(2*k-1-t)*s+j]
            _s += a*b

        s3 += t * _s

    return s1 + s2 + s3


def big_sigma(b_matrix, phi, t: int, s: int):
    a = 2 / (t + 1)

    sigm = 0

    for r in range(1, t):
        _s = 0
        for j in range(1, 2*s-1):
            b = b_matrix[s][j]
            c = phi[(2*s-1)*s+j]
            _s = b * c

        sigm += r * _s

    return a * sigm


def _2ks_iter(phi, b_matrix, s, k, lim_0, lim_1):
    # setup y
    y = np.zeros(shape=2*s*k+1, dtype=float)
    y[0] = lim_0
    y[-1] = lim_1
    mid_point = 0.5 * (lim_0 + lim_1) + sigma(b_matrix, phi, k, s)
    y[k*s] = mid_point

    # backwards propagate odd nodes
    for t in range(2*k-1, 0, -1):
        y[t*s] = t/(t+1)*y[(t+1)*s] + 1/(t+1)*lim_0 + big_sigma(b_matrix, phi, t, s)

    # forwards propagate even nodes
    for t in range(1, 2*k-1):
        for i in range(2, s+2):
            # for odd nodes skip as they are already propagated
            if ((t-1)*s+i) % 2 == 1:
                continue

            # skip out of bounds
            if (t == 2*k - 2) and (i == s+1):
                continue

            a = (2*s - (i-1)) / (2*s)
            b = y[(t-1)*s]

            c = (i-1)/(2*s)
            d = y[(t+1)*s]

            e = 0
            for j in range(2, 2*s):
                e += b_matrix[i-1][j-1]*phi[(t-1)*s+j-1]

            y[(t-1)*s+i-1] = a*b + c*d + e

    return y


RHS: TypeAlias = Callable[[float, float], float]


def two_ks_method_approx(
    rhs: RHS,
    number_of_iterations=1,
    s=2,
    k=4,
    lim_0=1/2,
    lim_1=1/4,
    verbose=True,
):
    initial_nodal_points = compute_initial_nodal_points(s)
    initial_approximation = partial(rhs, 0)
    h = 1/(2*k*s)

    phi = np.array([initial_approximation(i*h) for i in range(2*s*k+1)])
    b_matrix = compute_b_matrix(2*s+1, initial_nodal_points)

    for iteration_step in range(number_of_iterations):
        y = _2ks_iter(
            phi, b_matrix, s, k, lim_0, lim_1
        )
        for i in range(2*s*k+1):
            phi[i] = rhs(y[i], i*h)

        if verbose:
            print("step :=", iteration_step)
            print("-" * 100)
            print("y :=", y)
            print("phi :=", phi)
            print("mid :=", phi[k*s])
            print("-" * 100)

    return phi


if __name__ == '__main__':
    def rhs(y, x):
        return 2*y**2/(1+x)

    two_ks_method_approx(rhs=rhs, number_of_iterations=5, s=2, k=4)
