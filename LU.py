import numpy as np
def lu_decomposition(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for j in range(n):
        L[j][j] = 1

        for i in range(j+1):
            s1 = sum(U[k][j] * L[i][k] for k in range(i))
            U[i][j] = A[i][j] - s1

        for i in range(j, n):
            s2 = sum(U[k][j] * L[i][k] for k in range(j))
            L[i][j] = (A[i][j] - s2) / U[j][j]

    return L, U

def solve_lu(L, U, B):
    n = len(B)
    Y = np.zeros(n)
    X = np.zeros(n)

    # Solve LY = B
    for i in range(n):
        Y[i] = B[i] - sum(L[i][j] * Y[j] for j in range(i))

    # Solve UX = Y
    for i in range(n-1, -1, -1):
        X[i] = (Y[i] - sum(U[i][j] * X[j] for j in range(i+1, n))) / U[i][i]

    return X