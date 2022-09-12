import numpy as np

def proj(x,y):
	y1=y/np.linalg.norm(y)
	return np.dot(x,y1)*y1

v=np.loadtxt('vecs.txt')

n=len(v)

u=np.copy(v)

for i in range(1,n):
	for j in range(i):
		u[i]=u[i]-proj(u[i],u[j])

print("Orthogonalized vectors:")
print(u)
#np.savetxt('vecsOrth.txt',u)
np.savetxt('vecsOrth.txt',u,fmt='%.8f',delimiter='\t')
print("Written to file\nChecking dot products:")
for i in range(n-1):
	for j in range(i+1,n):
		print(i,j,np.dot(u[i],u[j]))
