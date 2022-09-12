import numpy as np


def gaussSeidel(A, b, tol):
    n=len(A)
    x = np.zeros_like(b)
    for k in range(maxit):
        x_new = np.zeros_like(x)
        #print("Iteration {0}: {1}".format(k, x))
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

def gaussSeidel_inverse(A,tol):
    n=len(A)
   
    Inv_A=[]
    for i in range(n):
        b = np.zeros_like(A[:,0])
        b[i]=1.0
        a=gaussSeidel(A, b, tol)
        Inv_A.append(a)
    return(np.transpose(np.asarray(Inv_A)))

def determinant_gauss_elemination(A):
    """
    Calculates the forward part of Gaussian elimination.
    """
    n=len(A)
    for k in range(0, n-1):
        for i in range(k+1, n):
            factor = A[i,k] / A[k,k]
            for j in range(k+1, n):
                A[i,j] = A[i,j]-factor*A[k,j]
    det=1
    for i in range(n):
        det=det*A[i,i]
    return det



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
    print("")
    x=gaussSeidel(A, b, tol)
   
    print("Solution: {0}".format(np.round(x,4)))
    print("")
    print("The inverse of the coefficient matrix is:")    
    Inv_A=gaussSeidel_inverse(A,tol)
    print(np.round(Inv_A,4))
    
    det_A=determinant_gauss_elemination(A)
    det_Inv_A=determinant_gauss_elemination(Inv_A)
    print("")
    print("The determinant of the co-efficient matrix = {0}  and".format(det_A))
    print("the determinant of the inverse of co-efficient matrix = {0}".format(np.round(det_Inv_A,4)))