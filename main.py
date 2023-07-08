from typing import Callable, TypeAlias

import numpy as np
from scipy.integrate import quad
from functools import partial


MA = 1  # 10 ** 5


def lagrangian_polynomial(
    p: int, j: int, nodal_points: list
) -> Callable[[float], float]:
    def sub_f(i: int, j: int):
        return lambda x: (x - nodal_points[i]) / (nodal_points[j] - nodal_points[i])

    def poly(x: float):
        prod = 1

        for i in range(1, p - 1):
            if i == j:
                continue

            prod *= sub_f(i, j)(x)

        return prod

    return poly


def compute_initial_nodal_points(s: int) -> list[int]:
    p = 2 * s + 1
    h = 1 / (p - 1)
    return np.fromfunction(lambda _, j: j * h, (1, p), dtype=float)[0]


def compute_b_matrix(s: int, k: int):
    p = 2 * s + 1
    nodal_points = compute_initial_nodal_points(s)

    b_matrix = np.zeros(shape=(p, p))
    b_matrix[0] = [None] * p
    b_matrix[-1] = [None] * p

    def integrand_of_poly(poly):
        return lambda x: quad(poly, 0, x)[0]

    a = 1 / nodal_points[p - 1]
    b = nodal_points[p - 1]

    for i in range(1, p - 1):
        b_matrix[i][0] = None
        b_matrix[i][p - 1] = None

        for j in range(1, p - 1):
            integrand = integrand_of_poly(lagrangian_polynomial(p, j, nodal_points))

            c = quad(integrand, 0, nodal_points[i])[0]
            d = nodal_points[i]
            e = quad(integrand, 0, nodal_points[p - 1])[0]

            b_matrix[i][j] = (a * (b * c - d * e)) / k**2

    return b_matrix


def compute_c_matrix(s: int, k: int):
    p = 2 * s + 1
    nodal_points = compute_initial_nodal_points(s)

    c_matrix = np.zeros(shape=(p, p))

    def integrand_of_poly(poly):
        return lambda x: quad(poly, nodal_points[i], x)[0]

    for i in range(0, p):
        for j in range(1, p - 1):
            integrand = integrand_of_poly(lagrangian_polynomial(p, j, nodal_points))

            full_integral = quad(integrand, 0, 1)[0]

            c_matrix[i][j] = full_integral

    return c_matrix / k**2


def sigma(b_matrix, phi, k: int, s: int):
    """computes sigma at ks+1"""

    s1 = 0
    for t in range(1, k):
        _s = 0
        for j in range(2, 2 * s + 1):
            a = b_matrix[s][j - 1]
            b = phi[(t - 1) * s + j - 1]
            _s += a * b

        s1 += t * _s
    print("s1", s1)

    s2 = 0
    for j in range(2, 2 * s + 1):
        a = b_matrix[s][j - 1]
        b = phi[(k - 1) * s + j - 1]
        s2 += a * b

    s2 = k * s2
    print("s2", s2)

    s3 = 0
    for t in range(1, k):
        _s = 0
        for j in range(2, 2 * s + 1):
            a = b_matrix[s][j - 1]
            b = phi[(2 * k - 1 - t) * s + j - 1]
            _s += a * b

        s3 += t * _s

    print("s3", s3)

    return s1 + s2 + s3


def big_sigma(b_matrix, phi, t: int, s: int):
    a = 2 / (t + 1)

    sigm = 0

    for r in range(1, t + 1):
        _s: int = 0
        for j in range(2, 2 * s + 1):
            b = b_matrix[s][j - 1]
            c = phi[(r - 1) * s + j - 1]
            _s = b * c

        sigm += r * _s

    return a * sigm


def big_sigma_2(b_matrix, phi, t: int, s: int, k: int):
    a = 2 / (t + 1)

    sigm = 0

    for r in range(1, t + 1):
        _s: int = 0
        for j in range(2, 2 * s + 1):
            b = b_matrix[s][j - 1]
            c = phi[(2 * k - r - 1) * s + j - 1]
            _s = b * c

        sigm += r * _s

    return a * sigm


def _2ks_iter(phi, b_matrix, s, k, lim_0, lim_1):
    # setup y
    y = np.zeros(shape=2 * s * k + 1, dtype=float)
    y[0] = lim_0
    y[-1] = lim_1
    y[k * s] = 0.5 * (lim_0 + lim_1) + sigma(b_matrix, phi, k, s)
    computed_nodes = [k * s]

    # backwards propagate odd nodes
    # [ <-- mid -- ]
    for t in range(k - 1, 0, -1):  # an(k, 1, -1)
        y[t * s] = (
            t / (t + 1) * y[(t + 1) * s]
            + 1 / (t + 1) * lim_0
            + big_sigma(b_matrix, phi, t, s)
        )
        computed_nodes.append(t * s)

    # forward propagate odd nodes [ -- mid --> ]
    for t in range(k - 1, 0, -1):
        y[(2 * k - t) * s] = (
            t / (t + 1) * y[(2 * k - t - 1) * s]
            + 1 / (t + 1) * lim_1
            + big_sigma_2(b_matrix, phi, t, s, k)
        )
        computed_nodes.append((2 * k - t) * s)

    # forwards propagate left over nodes
    for t in range(1, 2 * k - 1):
        for i in range(2, s + 2):
            # for odd nodes skip as they are already propagated
            if ((t - 1) * s + i - 1) in computed_nodes:
                continue

            a = (2 * s - (i - 1)) / (2 * s)
            b = y[(t - 1) * s]

            c = (i - 1) / (2 * s)
            d = y[(t + 1) * s]

            e = 0
            for j in range(1, 2 * s):
                e += b_matrix[i - 1][j] * phi[(t - 1) * s + j - 1]

            y[(t - 1) * s + i - 1] = a * b + c * d + e

    # CASE WHEN T = 2K - 1
    t = 2 * k - 1
    for i in range(2, 2 * s + 1):
        # for odd nodes skip as they are already propagated
        if ((t - 1) * s + i - 1) in computed_nodes:
            continue

        a = (2 * s - (i - 1)) / (2 * s)
        b = y[(t - 1) * s]

        c = (i - 1) / (2 * s)
        d = y[(t + 1) * s]

        e = 0
        for j in range(1, 2 * s):
            e += b_matrix[i - 1][j] * phi[(t - 1) * s + j - 1]

        y[(t - 1) * s + i - 1] = a * b + c * d + e

    return y


def compute_differentials(
    s: int,
    k: int,
    lim_0: float,
    lim_1: float,
    phi,
    b_matrix,
    c_matrix,
):
    y_diff = np.array([np.nan for _ in range(0, 2 * s * k + 2)])

    sum_1 = 0
    for r in range(1, k):
        _s = 0
        for j in range(2, 2 * s + 1):
            _s += b_matrix[s][j - 1] * (
                phi[(2 * k - r - 1) * s + j - 1] - phi[(r - 1) * s + j - 1]
            )
        sum_1 += float(r) * _s

    for i in range(1, 2 * s + 2):
        sum_2 = 0
        for j in range(2, 2 * s + 1):
            sum_2 += c_matrix[i - 1][j - 1] * phi[(k - 1) * s + j - 1]

        sum_2 = float(k) * sum_2
        y_diff[(k - 1) * s + i - 1] = lim_1 - lim_0 + 2 * sum_1 - sum_2

    return newton_cotes(k, s, y_diff)


def newton_cotes(k: int, s: int, y_diff):
    h = 1 / (2 * k * s)

    def newton_4(f1, f2, f3, f4):
        return 5 / 24 * h * (11 * f1 + f2 + f3 + 11 * f4)

    right = (k + 1) * s
    left = (k - 1) * s

    # Right propagate --> I_x_4^x_8 = y(x_8) =  y(x_4) + newton
    for i in range(right + 1, 2 * k * s + 2):
        y_diff[i] = y_diff[i - 4] + newton_4(
            y_diff[i - 4], y_diff[i - 3], y_diff[i - 2], y_diff[i - 1]
        )

    # Left propagate <-- I_x_3^x_7 = y(x_3) =  y(x_7) - newton
    for i in range(left - 1, -1, -1):
        y_diff[i] = y_diff[i + 4] - newton_4(
            y_diff[i + 4], y_diff[i + 3], y_diff[i + 2], y_diff[i + 1]
        )

    return y_diff


RHS: TypeAlias = Callable[[float, float, float], float]
Sol: TypeAlias = Callable[[float], float]


def two_ks_method_approx(
    rhs: RHS,
    lim_0: float,
    lim_1: float,
    number_of_iterations: int,
    s: int,
    k: int,
    u: Sol,
    u_diff: Sol,
    verbose: bool = True,
):
    b_matrix = compute_b_matrix(s, k)
    print("b_matrix :=", b_matrix)

    c_matrix = compute_c_matrix(s, k)
    print("c_matrix :=", c_matrix)

    initial_approximation = partial(rhs, 1, 1)
    h = 1 / (2 * k * s)

    phi = np.array([initial_approximation(i * h) for i in range(2 * s * k + 1)])
    y = np.array([initial_approximation(i * h) for i in range(2 * s * k + 1)])
    y_diff = np.array([u_diff(0) for _ in range(2*s*k+2)])
    print("initial_phi := ", phi)

    for iteration_step in range(1, number_of_iterations + 1):
        y = _2ks_iter(phi, b_matrix, s, k, lim_0, lim_1)
        for i in range(2 * s * k + 1):
            phi[i] = rhs(y[i], i * h, y_diff[i])
        y_diff = compute_differentials(s, k, lim_0, lim_1, phi, b_matrix, c_matrix)

        if verbose:
            error = get_s_errror(y, s=S, k=K, sol=u)
            print("-" * 100)
            print("step :=", iteration_step)
            print("y :=", y)
            print("phi :=", phi)
            print("mid :=", phi[k * s])
            print("error :=", error)
            print("max :=", np.max(error))

        print("diffs := ", y_diff)
        print("diff error:= ", get_diff_errror(y_diff, k, s, sol=u_diff))
        print("diff max:= ", np.average(get_diff_errror(y_diff, k, s, sol=u_diff)))

    return y


def get_s_errror(y, k, s, sol: Sol):
    h = 1 / (2 * k * s)
    solution = np.fromfunction(lambda _, i: sol(h * i), (1, 2 * k * s), dtype=float)[0]

    return np.abs(y[1 : 2 * k * s] - solution[1 : 2 * k * s]) / MA


def get_diff_errror(y, k, s, sol: Sol):
    h = 1 / (2 * k * s)
    solution = np.fromfunction(
        lambda _, i: sol(h * i), (1, 2 * k * s + 2), dtype=float
    )[0]
    return np.abs(y[0 : 2 * k * s + 2] - solution[0 : 2 * k * s + 2]) / MA


if __name__ == "__main__":

    def rhs(y, x, y_d):
        return x - y_d + x**2 / 2 - 1 / 6

    def rhs_2(y, x, y_d):
        return -y * y_d + y**3

    def u2(x):
        return 1/(x+4)

    def u2_diff(x):
        return -(1/(x+4))**2

    S = 2
    K = 2

    def u(x):
        return 1 / 6 * x**3 - 1 / 6 * x

    def u_diff(x):
        return 1 / 2 * x**2 - 1 / 6

    y = two_ks_method_approx(
        rhs=rhs,
        lim_0=0,
        lim_1=0,
        number_of_iterations=3,
        s=S,
        k=K,
        u=u,
        u_diff=u_diff,
        verbose=False
    )
