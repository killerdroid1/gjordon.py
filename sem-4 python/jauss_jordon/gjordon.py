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

def Gauss_Jordon(A, b):
    """
    Calculates the forward part of Gaussian elimination.
    """
    n=len(A)
    for k in range(0, n):
        pivot(n,A,b,k)
        for i in range(n):
            if k!=i:
                factor = A[i,k] / A[k,k]
                print(factor)
                for j in range(n):
                    A[i,j] = A[i,j]-factor*A[k,j]
                b[i] = b[i] - factor * b[k]

    print('A = \n%s and b = %s' % (A,b))
    x = np.zeros((n,1))
    for k in range(n):
        x[k] = b[k] / A[k,k]
    return x


# Main program starts here
if __name__ == '__main__':
    A = np.array([[3,6,1],
                  [2,4,3],
                  [1,3,2]],dtype=float)
    b = np.array([16,13,9],dtype=float)
    x = Gauss_Jordon(A, b)
    print('Gauss result is x = \n %s' % x)