import math


# -----------------------------------------------------------
# Функция варианта №7
# -----------------------------------------------------------
def f(x):
    return math.sqrt(1 + 2 * x**3)


# -----------------------------------------------------------
# Метод трапеций
# -----------------------------------------------------------
def trapezoid(a, b, N):
    h = (b - a) / N
    s = 0.5 * (f(a) + f(b))
    for i in range(1, N):
        s += f(a + i * h)
    return s * h


# -----------------------------------------------------------
# Метод Симпсона
# -----------------------------------------------------------
def simpson(a, b, N):
    # N должно быть четным
    if N % 2 == 1:
        N += 1

    h = (b - a) / N
    s = f(a) + f(b)

    for i in range(1, N):
        x = a + h * i
        if i % 2 == 0:
            s += 2 * f(x)
        else:
            s += 4 * f(x)

    return s * h / 3


# -----------------------------------------------------------
# Алгоритм с сгущающейся сеткой (формула 5.4 или 5.6)
# -----------------------------------------------------------
def integrate_with_refinement(method, a, b, eps, criterion_type=1):
    """
    method — trapezoid или simpson
    criterion_type:
        1 → критерий формулы (5.4): |I₂N − IN| ≤ eps
        2 → критерий формулы (5.6): |I₂N − IN| ≤ 3eps
    """

    N = 4
    I_prev = method(a, b, N)

    while True:
        N *= 2
        I_new = method(a, b, N)

        diff = abs(I_new - I_prev)

        if criterion_type == 1:
            # критерий (5.4)
            if diff <= eps:
                return I_new, N, diff
        else:
            # критерий (5.6)
            if diff <= 3 * eps:
                return I_new, N, diff

        I_prev = I_new


# -----------------------------------------------------------
# Основная программа
# -----------------------------------------------------------
if __name__ == "__main__":
    a = 1.2
    b = 2.471
    eps_values = [1e-4, 1e-5]

    print("Лабораторная работа №5 — Численное интегрирование")
    print("Вариант №7: f(x) = sqrt(1 + 2x^3), интервал [1.2; 2.471]\n")

    for eps in eps_values:
        print(f"==========================")
        print(f"Точность eps = {eps}")
        print(f"==========================\n")

        # --- Метод трапеций ---
        I_trap_54, N_trap_54, diff54 = integrate_with_refinement(trapezoid, a, b, eps, criterion_type=1)
        I_trap_56, N_trap_56, diff56 = integrate_with_refinement(trapezoid, a, b, eps, criterion_type=2)

        # --- Метод Симпсона ---
        I_simp_54, N_simp_54, diffS54 = integrate_with_refinement(simpson, a, b, eps, criterion_type=1)
        I_simp_56, N_simp_56, diffS56 = integrate_with_refinement(simpson, a, b, eps, criterion_type=2)

        # ------------------------------------
        # Вывод результатов
        # ------------------------------------
        print("Метод трапеций:")
        print(f"  Критерий (5.4): I = {I_trap_54:.10f},  N = {N_trap_54},  |I₂N − IN| = {diff54:.2e}")
        print(f"  Критерий (5.6): I = {I_trap_56:.10f},  N = {N_trap_56},  |I₂N − IN| = {diff56:.2e}\n")

        print("Метод Симпсона:")
        print(f"  Критерий (5.4): I = {I_simp_54:.10f},  N = {N_simp_54},  |I₂N − IN| = {diffS54:.2e}")
        print(f"  Критерий (5.6): I = {I_simp_56:.10f},  N = {N_simp_56},  |I₂N − IN| = {diffS56:.2e}")

        print("\n--------------------------------------\n")
