import numpy as np

def proj(x,y):
	y1=y/np.linalg.norm(y)
	return np.dot(x,y1)*y1

v=np.array([[1,2,3,5],[2,3,3,1],[1,1,1,1],[7,5,9,1],[3,1,3,1]],dtype=float)
#v=np.array([[1,2],[3,1]],dtype=float)

print("The chosen input vectors are (row-wise):")
print(v)


n=len(v)

u=np.copy(v)

for i in range(1,n):
	for j in range(i):
		u[i]=u[i]-proj(u[i],u[j])

print("Orthogonalized vectors:")
print(u)
print("\nChecking dot products:")
for i in range(n-1):
	for j in range(i+1,n):
		print(i,j,np.round(np.dot(u[i],u[j]),2))
