import numpy as np

def pivot(n,a,b,k):
    p=k
    large=abs(a[k,k])
    for i in range(k+1,n):
        if abs(a[i,k])>large:
            large=abs(a[i,k])
            p=i
        print(a)
    if p!=k:
        a[[p,k]]=a[[k,p]]
        b[k], b[p] = b[p], b[k]
    return a,b,k

def forward_elimination(A, b, n):
    """
    Calculates the forward part of Gaussian elimination.
    """
    for k in range(0, n-1):
        pivot(n,A,b,k)
        for i in range(0, n):
            if i!=k:
                factor = A[i,k] / A[k,k]
                print(factor)
                for j in range(k+1, n):
                    A[i,j] = A[i,j]-factor*A[k,j]
                b[i] = b[i] - factor * b[k]

        print('A = \n%s and b = %s' % (A,b))
    return A, b

def back_substitution(a, b, n):
    """"
    Does back substitution, returns the Gauss result.
    """
    for k in range(n):
        x[k] = b[k] / a[k,k]
    return x

def gauss(A, b):
    """
    This function performs Gauss elimination without pivoting.
    """
    n = len(A)

  
    A, b = forward_elimination(A, b, n)
    return back_substitution(A, b, n)

# Main program starts here
if __name__ == '__main__':
    A = np.array([[3,6,1],
                  [2,4,3],
                  [1,3,2]],dtype=float)
    b = np.array([16,13,9],dtype=float)
    x = gauss(A, b)
    print('Gauss result is x = \n %s' % x)