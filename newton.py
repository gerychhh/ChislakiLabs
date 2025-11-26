import numpy as np
from Gauss import gauss_elimination

def F(x):
    x1, x2 = x
    f1 = np.cos(0.4 * x2 + x1**2) + x2**2 + x1**2 - 1.6
    f2 = 1.5 * x1**2 - x2**2 / 0.36 - 1
    return np.array([f1, f2])


def jacobian_analytic(x):
    x1, x2 = x
    return np.array([
        [-2 * x1 * np.sin(0.4 * x2 + x1**2) + 2 * x1,
         -0.4 * np.sin(0.4 * x2 + x1**2) + 2 * x2],
        [3 * x1, -2 * x2 / 0.36]
    ])


def jacobian_numeric(f, x, M=0.01):
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        dx[j] = M * x[j] if x[j] != 0 else M
        J[:, j] = (f(x + dx) - f(x - dx)) / (2 * dx[j])
    return J


def newton_method(F, x0, eps1=1e-9, eps2=1e-9, NIT=100, method="analytic", M=0.01):
    x = x0.astype(float)

    print(f"\nМетод Ньютона ({method})")
    print("k | δ1          | δ2")
    print("-" * 26)

    for k in range(NIT):
        Fx = F(x)

        if method == "analytic":
            J = jacobian_analytic(x)
        elif method == "numeric":
            J = jacobian_numeric(F, x, M)
        else:
            raise ValueError("method должен быть 'analytic' или 'numeric'")

        delta_x = gauss_elimination(J.copy(), -Fx.copy())
        x_new = x + delta_x

        delta1 = np.max(np.abs(Fx))

        step = np.abs(delta_x)
        scale = np.where(np.abs(x_new) >= 1.0, np.abs(x_new), 1.0)
        delta2 = np.max(step / scale)

        print(f"{k+1:2d} | {delta1: .3e} | {delta2: .3e}")

        if delta1 < eps1 and delta2 < eps2:
            print(f"\nРешение найдено за {k+1} итераций:\n{x_new}\n")
            return x_new, k + 1

        x = x_new

    print(f"\nПревышено число итераций ({NIT})")
    return x, NIT


if __name__ == "__main__":
    x0 = np.array([1.0, -1.0])
    newton_method(F, x0, method="analytic")

    for M in [0.01, 0.001, 0.0001]:
        newton_method(F, x0, method="numeric", M=M)
