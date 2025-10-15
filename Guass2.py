def gauss_elimination(A, b):
    n = len(b)
    A = [row[:] for row in A]
    b = b[:]
    for k in range(n - 1):
        max_row = max(range(k, n), key=lambda i: abs(A[i][k]))
        if abs(A[max_row][k]) < 1e-18:
            raise ValueError("Матрица вырождена")
        if max_row != k:
            A[k], A[max_row] = A[max_row], A[k]
            b[k], b[max_row] = b[max_row], b[k]
        for i in range(k + 1, n):
            factor = A[i][k] / A[k][k]
            for j in range(k, n):
                A[i][j] -= factor * A[k][j]
            b[i] -= factor * b[k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - s) / A[i][i]
    return x


def residual(A, x, b):
    n = len(A)
    r = [sum(A[i][j] * x[j] for j in range(n)) - b[i] for i in range(n)]
    norm_inf = max(abs(ri) for ri in r)
    return r, norm_inf


def relative_error(x_ref, x_test):
    num = max(abs(x_ref[i] - x_test[i]) for i in range(len(x_ref)))
    den = max(abs(xi) for xi in x_test)
    return num / den if den != 0 else float('inf')


def fmt_vec(v, prec=6):
    return "[" + ", ".join(f"{val:.{prec}g}" for val in v) + "]"


if __name__ == "__main__":
    tasks = {
        1: (
            [[6, 13, -17],
             [13, 29, -38],
             [-17, -38, 50]],
            [2, 4, -5]
        ),
        2: (
            [[1, 2, 1],
             [-1, -2, 2],
             [0, 1, 1]],
            [1, 1, 2]
        ),
        3: (
            [[2.30, 5.70, -0.80],
             [3.50, -2.70, 5.30],
             [1.70, 2.30, -1.80]],
            [-6.49, 19.20, -5.09]
        ),
        7: (
            [[2.60, -4.50, -2.00],
             [3.00,  3.00,  4.30],
             [-6.00, 3.50,  3.00]],
            [19.07, 3.21, -18.25]
        )
    }

    num = int(input("Вариант условия (1, 2, 3, 7 или 21): "))

    if num == 21:
        λ1 = float(input("Введите λ1: "))
        λ2 = float(input("Введите λ2: "))
        λ3 = float(input("Введите λ3: "))
        A = [
            [2*λ1 + 4*λ2,      2*(λ1 - λ2),      2*(λ1 - λ2)],
            [2*(λ1 - λ2),      2*λ1 + λ2 + 3*λ3, 2*λ1 + λ2 - 3*λ3],
            [2*(λ1 - λ2),      2*λ1 + λ2 - 3*λ3, 2*λ1 + λ2 + 3*λ3]
        ]
        b = [
            -4*λ1 - 2*λ2,
            -4*λ1 + λ2 + 9*λ3,
            -4*λ1 + λ2 - 9*λ3
        ]
    elif num in tasks:
        A, b = tasks[num]
    else:
        print("Нет такой задачи!")
        exit()

    x = gauss_elimination(A, b)
    r, norm_r = residual(A, x, b)
    b_aux = [sum(A[i][j] * x[j] for j in range(3)) for i in range(3)]
    x_aux = gauss_elimination(A, b_aux)
    rel_err = relative_error(x, x_aux)
    print(f"\nвариант: {num} ")
    print("x =", fmt_vec(x))
    print("невязка =", fmt_vec(r))
    print("||r|| =", f"{norm_r:.3e}")
    print("x_aux =", fmt_vec(x_aux))
    print("дельта =", f"{rel_err:.3e}")
