import numpy as np


def gaussSeidel(A, b, tol):
    n=len(A)
    x = np.zeros_like(b)
    for k in range(maxit):
        x_new = np.zeros_like(x)
        print("Iteration {0}: {1}".format(k, x))
        for i in range(n):
            summ = 0.0
            for j in range(n):
                if (j != i):
                    summ = summ + A[i][j] * x[j]
            x_new[i] = (b[i] - summ) / A[i][i]
        if np.allclose(x, x_new, rtol=tol):
                break
        x = x_new
    return(x)

# Main program starts here
if __name__ == '__main__':
    tol=0.00001
    maxit = 1000

    # initialize the matrix
    A = np.array([[2., 1., 1.],
                  [3., 5., 2.],
                  [2., 1., 4.]])
    # initialize the RHS vector
    b = np.array([5., 15., 8.])

    print("System of equations:")
    for i in range(A.shape[0]):
        row = ["{0:3g}*x{1}".format(A[i, j], j + 1) for j in range(A.shape[1])]
        print("[{0}] = [{1:3g}]".format(" + ".join(row), b[i]))

    x=gaussSeidel(A, b, tol)

    print("Solution: {0}".format(x))