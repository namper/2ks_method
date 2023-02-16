from typing import Callable, TypeAlias

import numpy as np
from scipy.integrate import quad
from functools import partial


def lagrangian_polynomial(p: int, j: int, nodal_points: list) -> Callable[[float], float]:

    def sub_f(i: int, j: int):
        return lambda x: (x-nodal_points[i])/(nodal_points[j] - nodal_points[i])

    def poly(x: float):
        prod = 1

        for i in range(1, p-1):
            if i == j:
                continue

            prod *= sub_f(i, j)(x)

        return prod

    return poly


def compute_initial_nodal_points(s: int) -> list[int]:
    p = 2*s+1
    h = 1/(p-1)
    return np.fromfunction(lambda _, j: j*h, (1, p), dtype=float)[0]


def compute_b_matrix(s: int, k: int):
    p = 2*s+1
    nodal_points = compute_initial_nodal_points(s)

    b_matrix = np.zeros(shape=(p, p))
    b_matrix[0] = [None] * p
    b_matrix[-1] = [None] * p

    def integrand_of_poly(poly):
        return lambda x: quad(poly, 0, x)[0]

    a = 1/nodal_points[p-1]
    b = nodal_points[p-1]

    for i in range(1, p-1):
        b_matrix[i][0] = None
        b_matrix[i][p-1] = None

        for j in range(1, p-1):
            integrand = integrand_of_poly(
                    lagrangian_polynomial(p, j, nodal_points)
            )

            c = quad(integrand, 0, nodal_points[i])[0]
            d = nodal_points[i]
            e = quad(integrand, 0, nodal_points[p-1])[0]

            b_matrix[i][j] = a * (b*c - d*e) / k**2

    return b_matrix


def sigma(b_matrix, phi, k: int, s: int):
    """computes sigma at ks+1 """

    s1 = 0
    for t in range(1, k):
        _s = 0
        for j in range(1, 2*s):
            a = b_matrix[s][j]
            b = phi[(t-1)*s+j]
            _s += a*b

        s1 += t * _s

    s2 = 0
    for j in range(1, 2*s):
        a = b_matrix[s][j]
        b = phi[(k-1)*s+j]
        s2 += a*b

    s2 = k * s2

    s3 = 0
    for t in range(1, k):
        _s = 0
        for j in range(1, 2*s):
            a = b_matrix[s][j]
            b = phi[(2*k-1-t)*s+j]
            _s += a*b

        s3 += t * _s

    return s1 + s2 + s3


def big_sigma(b_matrix, phi, t: int, s: int):
    a = 2 / (t + 1)

    sigm = 0

    for r in range(1, t+1):
        _s: int = 0
        for j in range(1, 2*s):
            b = b_matrix[s][j]
            c = phi[(r-1)*s+j]
            _s = b * c

        sigm += r * _s

    return a * sigm


def big_sigma_2(b_matrix, phi, t: int, s: int, k: int):
    a = 2 / (t + 1)

    sigm = 0

    for r in range(1, t+1):
        _s: int = 0
        for j in range(1, 2*s):
            b = b_matrix[s][j]
            c = phi[(2*k-r-1)*s+j]
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

    computed_nodes = [k*s]

    # backwards propagate odd nodes
    # [ <-- mid -- ]
    for t in range(k-1, 0, -1):
        y[t*s] = t/(t+1)*y[(t+1)*s] + 1/(t+1)*lim_0 + big_sigma(b_matrix, phi, t, s)
        computed_nodes.append(t*s)

    # forward propagate odd nodes [ -- mid --> ]
    for t in range(k-1, 0, -1):
        y[(2*k-t)*s] = t/(t+1)*y[(2*k-t-1)*s] + 1/(t+1)*lim_1 + big_sigma_2(b_matrix, phi, t, s, k)
        computed_nodes.append((2*k-t+1)*s)

    # forwards propagate left over nodes
    for t in range(1, 2*k-1):
        for i in range(2, s+2):

            # for odd nodes skip as they are already propagated
            if ((t-1)*s+i-1) in computed_nodes:
                continue

            a = (2*s - (i-1)) / (2*s)
            b = y[(t-1)*s]

            c = (i-1)/(2*s)
            d = y[(t+1)*s]

            e = 0
            for j in range(1, 2*s):
                e += b_matrix[i-1][j]*phi[(t-1)*s+j-1]

            y[(t-1)*s+i-1] = a*b + c*d + e

    # CASE WHEN T = 2K - 1
    t = 2*k-1
    for i in range(2, 2*s+1):
        # for odd nodes skip as they are already propagated
        if ((t-1)*s+i-1) in computed_nodes:

            continue

        a = (2*s - (i-1)) / (2*s)
        b = y[(t-1)*s]

        c = (i-1)/(2*s)
        d = y[(t+1)*s]

        e = 0
        for j in range(1, 2*s):
            e += b_matrix[i-1][j]*phi[(t-1)*s+j-1]

        y[(t-1)*s+i-1] = a*b + c*d + e

    return y


RHS: TypeAlias = Callable[[float, float], float]


def two_ks_method_approx(
    rhs: RHS,
    lim_0,
    lim_1,
    number_of_iterations,
    s,
    k,
    verbose=True,
):
    b_matrix = compute_b_matrix(s, k)
    print("b_matrix :=", b_matrix)

    initial_approximation = partial(rhs, 0)
    h = 1/(2*k*s)

    phi = np.array([initial_approximation(i*h) for i in range(2*s*k+1)])

    y = np.array([initial_approximation(i*h) for i in range(2*s*k+1)])
    print("initial_phi := ", phi)

    for iteration_step in range(1, number_of_iterations+1):
        y = _2ks_iter(
            phi, b_matrix, s, k, lim_0, lim_1
        )
        for i in range(2*s*k+1):
            phi[i] = rhs(y[i], i*h)

        if verbose:
            error = get_abs_s_errror(y, s=S, k=K)

            print("-" * 100)
            print("step :=", iteration_step)
            print("y :=", y)
            print("phi :=", phi)
            print("mid :=", phi[k*s])
            print("error :=", error)

    return y


def get_abs_s_errror(y, k, s):

    def sol(x):
        return 1/(1+x)

    h = 1/(2*k*s)
    solution = np.fromfunction(
        lambda _, i: sol(h*i),
        (1, 2*k*s),
        dtype=float
    )[0]

    return np.average(np.abs(y[1:2*k*s] - solution[1:2*k*s]))


if __name__ == '__main__':
    def rhs(y, x):
        return 2*y**2/(1+x)

    S = 1
    K = 20

    y = two_ks_method_approx(
        rhs=rhs,
        lim_0=1,
        lim_1=1/2,
        number_of_iterations=20,
        s=S,
        k=K,
    )
