import numpy as np

# Exemplo de sistema de equações lineares bem condicionado
A = np.array([[20, 1, 1],  # 20x + y + z = 20
              [1, 15, 1],  # x + 15y + z = 10
              [1, 1, 25]])  # x + y + 25z = 15

b = np.array([20, 10, 15])
x0 = b / np.diag(A)  # Solução inicial modificada

def jacobi(A, b, x0, max_iter=1000, tol=1e-6):
    n = len(b)
    x = np.copy(x0)
    for _ in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if np.linalg.norm(x_new - x) < tol:
            return x_new
        x = np.copy(x_new)
    print("O método de Jacobi não convergiu após", max_iter, "iterações.")
    return x

# Resolver o sistema usando o método de Jacobi
solution = jacobi(A, b, x0)
print("Solução encontrada pelo método de Jacobi:", solution)