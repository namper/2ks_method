from typing import Callable
import numpy as np
from scipy.integrate import quad


def lagrangian_polynomial(i: int, nodal_points: np.ndarray) -> Callable[[float], float]:
    # @TODO: WRITE
    return lambda x: x


def compute_initial_nodal_points(s: int) -> list[int]:
    p = 2*s+1
    h = 1/(p-1)

    return np.fromfunction(lambda _, j: j*h, (1, p), dtype=float)


def compute_b_matrix(p: int, nodal_points: np.ndarray):

    b_matrix = np.zeros(shape=(p, p))

    def integrand_of_poly(poly):
        return lambda x: quad(poly, 0, x)[0]

    a = 1/nodal_points[p-1]
    b = nodal_points[p-1]

    for i in range(p):
        for j in range(p):
            integrand = integrand_of_poly(
                    lagrangian_polynomial(j, nodal_points)
            )

            c = quad(integrand, 0, nodal_points[i-1])[0]
            d = nodal_points[i]
            e = quad(integrand, 0, nodal_points[p-1])[0]

            b_matrix[i][j] = a * (b*c - d*e)

    return b_matrix


def approximate(order=2, initial_phi=0, s=2, k=4):
    pass


if __name__ == '__main__':
    approximate(order=2, initial_phi=0, s=2, k=4)
