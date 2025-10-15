import math

# --- функции системы ---
def F(x1, x2):
    f1 = math.cos(0.4 * x2 + x1**2) + x2**2 + x1**2 - 1.6
    f2 = 1.5 * x1**2 - (x2**2) / 0.36 - 1
    return [f1, f2]

# --- Якоби ---
def J(x1, x2):
    j11 = -2 * x1 * math.sin(0.4 * x2 + x1**2) + 2 * x1
    j12 = -0.4 * math.sin(0.4 * x2 + x1**2) + 2 * x2
    j21 = 3 * x1
    j22 = -2 * x2 / 0.36
    return [[j11, j12],
            [j21, j22]]

# --- Решение 2×2 системы линейных уравнений ---
def solve2x2(A, b):
    a11, a12 = A[0]
    a21, a22 = A[1]
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-15:
        raise ValueError("Определитель матрицы Якоби близок к нулю!")
    x1 = (b[0] * a22 - b[1] * a12) / det
    x2 = (a11 * b[1] - a21 * b[0]) / det
    return [x1, x2]

# --- Метод Ньютона ---
def newton(x1_0, x2_0, eps1=1e-9, eps2=1e-9, max_iter=100):
    x1, x2 = x1_0, x2_0

    for k in range(1, max_iter + 1):
        f = F(x1, x2)
        A = J(x1, x2)
        dx1, dx2 = solve2x2(A, [-f[0], -f[1]])
        x1_new = x1 + dx1
        x2_new = x2 + dx2

        delta1 = max(abs(f[0]), abs(f[1]))
        delta2 = max(abs(x1_new - x1) / (abs(x1_new) + eps2),
                     abs(x2_new - x2) / (abs(x2_new) + eps2))

        print(f"Итерация {k:2d}: x1={x1_new:.6f}, x2={x2_new:.6f}, δ1={delta1:.2e}, δ2={delta2:.2e}")

        if delta1 < eps1 and delta2 < eps2:
            print("\nРешение найдено за", k, "итераций.")
            return x1_new, x2_new

        x1, x2 = x1_new, x2_new

    print("\nПревышено максимальное число итераций!")
    return x1, x2

# --- Основная часть ---
print("Начальное приближение (1; -1):")
res1 = newton(1, -1)
print("Решение:", res1)

print("\nНачальное приближение (-1; 1):")
res2 = newton(-1, 1)
print("Решение:", res2)
