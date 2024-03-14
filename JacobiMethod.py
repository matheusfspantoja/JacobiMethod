import numpy as np

def input_matrix(prompt, size):
    print(prompt)
    matrix = []
    for i in range(size):
        row = list(map(float, input(f"Digite os valores da linha {i+1} separados por espaço: ").split()))
        matrix.append(row)
    return np.array(matrix)

def input_vector(prompt, size):
    print(prompt)
    values = list(map(float, input("Digite os valores separados por espaço: ").split()))
    return np.array(values)

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

# Entrada de dados para a matriz A
size_A = int(input("Digite o tamanho da matriz A: "))
A = input_matrix("Digite os valores da matriz A:", size_A)

# Entrada de dados para o vetor b
size_b = int(input("Digite o tamanho do vetor b: "))
b = input_vector("Digite os valores do vetor b:", size_b)

x0 = b / np.diag(A)  # Solução inicial modificada

# Resolver o sistema usando o método de Jacobi
solution = jacobi(A, b, x0)
print("Solução encontrada pelo método de Jacobi:", solution)
