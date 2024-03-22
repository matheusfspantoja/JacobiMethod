import numpy as np

def input_matrix(prompt, size):
    matrix = []
    for i in range(size):
        while True:
            try:
                row = list(map(float, input(f"Digite os valores da linha {i+1} separados por espaço: ").split()))
                if len(row) != size:
                    raise ValueError("O número de valores na linha {} não corresponde ao tamanho da matriz A.".format(i+1))
                matrix.append(row)
                break
            except ValueError:
                print("Por favor, insira apenas números válidos.")
    return np.array(matrix)

def input_vector(prompt, size):
    while True:
        try:
            values = list(map(float, input("Digite os valores do vetor b separados por espaço: ").split()))
            if len(values) != size:
                raise ValueError("O número de valores no vetor b não corresponde ao número de linhas da matriz A.")
            break
        except ValueError:
            print("Por favor, insira apenas números válidos.")
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
    raise Exception("O método de Jacobi não convergiu após {} iterações.".format(max_iter))

try:
    # Entrada de dados para a matriz A
    size_A = int(input("Digite o tamanho da matriz A: "))
    A = input_matrix("Digite os valores da matriz A:", size_A)

    # Entrada de dados para o vetor b
    b = input_vector("Digite os valores do vetor b:", size_A)

    x0 = b / np.diag(A)  # Solução inicial modificada

    # Resolver o sistema usando o método de Jacobi
    solution = jacobi(A, b, x0)
    print("Solução encontrada pelo método de Jacobi:", solution)
except ValueError as ve:
    print(ve)
except Exception as e:
    print(e)