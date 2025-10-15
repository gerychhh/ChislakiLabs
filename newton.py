import numpy as np

def F(x):
    x1, x2 = x
    f1 = np.cos(0.4 * x2 + x1**2) + x2**2 + x1**2 - 1.6
    f2 = 1.5 * x1**2 - x2**2 / 0.36 - 1
    return np.array([f1, f2])

def J(x):
    x1, x2 = x
    j11 = -2 * x1 * np.sin(0.4 * x2 + x1**2) + 2 * x1
    j12 = -0.4 * np.sin(0.4 * x2 + x1**2) + 2 * x2
    j21 = 3 * x1
    j22 = -2 * x2 / 0.36
    return np.array([[j11, j12],
                     [j21, j22]])

def newton(F, J, x0, eps1=1e-9, eps2=1e-9, max_iter=100):
    x = np.array(x0, dtype=float)
    for k in range(max_iter):
        Fx = F(x)
        Jx = J(x)
        delta = np.linalg.solve(Jx, -Fx)
        x_new = x + delta

        delta1 = np.max(np.abs(Fx))
        delta2 = np.max(np.abs(x_new - x) / (np.abs(x_new) + eps2))

        print(f"Итерация {k+1}: x = {x_new}, δ1 = {delta1:.2e}, δ2 = {delta2:.2e}")

        if delta1 < eps1 and delta2 < eps2:
            print("\nРешение найдено:")
            return x_new

        x = x_new
    print("\nПревышено максимальное количество итераций!")
    return x

print("Начальное приближение (1; -1)")
x1 = newton(F, J, [1, -1])
print("Решение:", x1)

print("\nНачальное приближение (-1; 1)")
x2 = newton(F, J, [-1, 1])
print("Решение:", x2)
