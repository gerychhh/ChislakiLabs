# lab4.py
# Лабораторная работа №4 — МНК, вариант 7
# Использует метод Гаусса из Gauss2.py

from Guass2 import gauss_elimination
import matplotlib.pyplot as plt
import numpy as np


def build_normal_system(x, y, m):
    """
    Строит нормальную систему МНК для полинома степени m:
        sum_j a_j * sum_i x_i^(j+k) = sum_i y_i * x_i^k
    Возвращает матрицу A и вектор b.
    """
    n = len(x)

    # Суммы степеней: powerx[k] = sum(x_i^k), k=0..2m
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
    """Возвращает значение полинома с коэффициентами a в точке x."""
    s = 0.0
    xn = 1.0
    for coef in a:
        s += coef * xn
        xn *= x
    return s


def main():
    # Таблица варианта 7
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
    m = 3  # степень полинома задаёт методичка

    # 1. Строим нормальные уравнения
    A, b = build_normal_system(t, C, m)

    # 2. Решаем их твоим методом Гаусса
    a = gauss_elimination(A, b)

    print("Коэффициенты аппроксимирующего полинома 3-й степени:")
    for i, ai in enumerate(a):
        print(f"a[{i}] = {ai:.10e}")
    print()

    # 3. Вывод таблицы и вычисление ошибок
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

    # Остаточная дисперсия
    r = n - (m + 1)
    S2 = sq_sum / r
    sigma = S2 ** 0.5

    print("\nОстаточная дисперсия S^2 =", f"{S2:.10e}")
    print("Среднеквадратическое отклонение sigma =", f"{sigma:.10e}")

    # ====== ГРАФИКИ ======

    # --- График аппроксимации ---
    t_dense = np.linspace(0, 100, 500)
    poly_vals = [poly_value(a, x) for x in t_dense]

    plt.figure()
    plt.scatter(t, C, label="Экспериментальные данные")
    plt.plot(t_dense, poly_vals, label="Полином 3-й степени")
    plt.xlabel("t, °C")
    plt.ylabel("C(t)")
    plt.title("Аппроксимация теплоёмкости воды")
    plt.grid(True)
    plt.legend()

    # --- График невязок ---
    plt.figure()
    plt.stem(t, residuals)   # без use_line_collection — для старого matplotlib
    plt.xlabel("t, °C")
    plt.ylabel("C(t) - P3(t)")
    plt.title("График невязок")
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
