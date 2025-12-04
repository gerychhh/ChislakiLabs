
from Guass2 import gauss_elimination
import matplotlib.pyplot as plt
import numpy as np


def build_normal_system(x, y, m):

    n = len(x)

    powerx = [sum(xi ** k for xi in x) for k in range(2 * m + 1)]

    # Матрица A
    A = [[0.0] * (m + 1) for _ in range(m + 1)]
    for l in range(m + 1):
        for j in range(m + 1):
            k = l + j
            A[l][j] = n if k == 0 else powerx[k]

    # Вектор b
    b = [sum(y[i] * (x[i] ** l) for i in range(n)) for l in range(m + 1)]

    return A, b


def poly_value(a, x):
    s = 0.0
    xn = 1.0
    for coef in a:
        s += coef * xn
        xn *= x
    return s


def main():
    t = [
        0.0, 5.0, 10.0, 15.0, 20.0, 25.0,
        30.0, 35.0, 40.0, 45.0, 50.0, 55.0,
        60.0, 65.0, 70.0, 75.0, 80.0, 85.0,
        90.0, 95.0, 100.0
    ]

    C = [
        1.00762, 1.00392, 1.00153, 1.00000, 0.99907, 0.99852,
        0.99826, 0.99818, 0.99828, 0.99849, 0.99878, 0.99919,
        0.99967, 1.00024, 1.00091, 1.00167, 1.00253, 1.00351,
        1.00461, 1.00586, 1.00721
    ]

    n = len(t)
    m = 5

    A, b = build_normal_system(t, C, m)

    a = gauss_elimination(A, b)

    print("Коэффициенты аппроксимирующего полинома 3-й степени:")
    for i, ai in enumerate(a):
        print(f"a[{i}] = {ai:.10e}")
    print()

    print("   t       C(t)          P3(t)        C(t)-P3(t)")
    print("--------------------------------------------------")

    sq_sum = 0.0
    residuals = []

    for ti, Ci in zip(t, C):
        Pi = poly_value(a, ti)
        diff = Ci - Pi
        residuals.append(diff)
        sq_sum += diff * diff

        print(f"{ti:6.1f}  {Ci:12.6f}  {Pi:12.6f}  {diff:12.6e}")

    r = n - (m + 1)
    S2 = sq_sum / r
    sigma = S2 ** 0.5

    print("\nОстаточная дисперсия S^2 =", f"{S2:.10e}")
    print("Среднеквадратическое отклонение sigma =", f"{sigma:.10e}")


    t_dense = np.linspace(0, 100, 500)
    poly_vals = [poly_value(a, x) for x in t_dense]
    plt.figure(figsize=(10, 6))
    plt.scatter(t, C, label="Экспериментальные данные f(x)", color='blue')
    plt.plot(t_dense, poly_vals, label="Аппроксимация φ(x) = P₃(t)", linewidth=2, color='red')

    plt.xlabel("t, °C")
    plt.ylabel("C(t)")
    plt.title("Графики функций f(x) и φ(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.show()


if __name__ == "__main__":
    main()
