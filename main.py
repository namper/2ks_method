from typing import Callable
import numpy as np
from scipy.integrate import quad


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


def rhs(y, x):
    return 2*y**2 / (1 + x)


def sigma(b_matrix, phi, k: int, s: int):
    """computes sigma at ks+1 """

    s1 = 0
    for t in range(0, k-2):
        _s = 0
        for j in range(1, 2*s-1):
            a = b_matrix[s][j-1]
            b = phi[(t-1)*s+j-1]
            _s += a*b

        s1 += t * _s

    s2 = 0
    for j in range(1, 2*s-1):
        a = b_matrix[s][j-1]
        b = 0  # @HACK: Needs to be fixed mid sum seems to be dependent on t but t is undefined here

        s2 += a*b

    s2 = k * s2

    s3 = 0
    for t in range(0, k-2):
        _s = 0
        for j in range(1, 2*s-1):
            a = b_matrix[s][j-1]
            b = phi[(2*k-1-t)*s+j-1]
            _s += a*b

        s3 += t * _s

    return s1 + s2 + s3


def approximate(order=2, initial_phi=0, s=2, k=4, lim_0=1/2, lim_1=1/4):
    p = 2*s+1
    initial_nodal_points = compute_initial_nodal_points(s)
    phi = np.array([initial_phi for _ in range(2*s*k+1)])
    b_matrix = compute_b_matrix(p, initial_nodal_points)

    mid_point = 0.5 * (lim_0 + lim_1) + sigma(b_matrix, phi, k, s)

    print("mid_point:=", mid_point)


if __name__ == '__main__':
    approximate(order=2, initial_phi=0, s=2, k=4)
