# #Allison Fenichel 
# #Homework4
# #EECS6892

import scipy
from scipy import io, stats
import pandas as pd
import numpy as np
from numpy.linalg import inv, pinv, det
import math
import matplotlib.pyplot as plt
import sys
import os



def GMM(K):
	dat=io.loadmat('data.mat')
	X=dat['X'].T
	N,d=X.shape
	T=100
	pi=np.ones((K))
	mu=scipy.random.random((K,d))
	Lamb=np.zeros((d,K,d))
	for j in range(K):
		Lamb[:,j,:]=np.diag(scipy.random.random(d))
	phi=scipy.random.random((N,K))
	n=np.zeros((K))	
	lnp=np.zeros(T)	
	for t in range(T):
		phi=e_step(t, K, X, mu, pi, Lamb, phi,n)
		n,mu,Lamb,pi=m_step(t,phi, K, X, mu, pi, Lamb, n)
		lnp[t]=likelihood(t, phi, K, X, mu, pi, Lamb, n)


	return phi,n,mu,Lamb,pi,lnp,X

def e_step(t, K, X, mu, pi, Lamb, phi, n):
	N,d=X.shape
	for i in range(N):
		denom=0
		for k in range(K):
			denom=denom+pi[k]*stats.multivariate_normal.pdf(X[i,:], mean=mu[k,:], cov=np.linalg.inv(Lamb[:,k,:]))
		for j in range(K):
			phi[i,j]=pi[j]*stats.multivariate_normal.pdf(X[i,:], mean=mu[j,:], cov=np.linalg.inv(Lamb[:,j,:]))/denom
	return phi		

def m_step(t,phi, K, X, mu, pi, Lamb, n):
	N,d=X.shape
	t=t+1
	for j in range(K):
		n[j]=np.sum(phi[:,j],0)
		mu[j,:]=1.0/n[j]*np.sum(np.multiply(phi[:,j].reshape(N,1),X),0)
		s=0
		for i in range(N):
			xminusmu=X[i,:]-mu[j,:]
			xminusmu=xminusmu.reshape(d,1)
			s=s+phi[i,j]*np.dot(xminusmu,xminusmu.T)	
		Lamb[:,j,:]=np.linalg.inv(1.0/n[j]*s)
		pi[j]=n[j]/N
	return n,mu,Lamb,pi	

def likelihood(t, phi, K, X, mu, pi, Lamb, n):
	t=t+1
	lnp=0
	N,d=X.shape
	for i in range(N):
		px=0
		for k in range(K):
			px=px+pi[k]*stats.multivariate_normal.pdf(X[i,:], mean=mu[k,:], cov=np.linalg.inv(Lamb[:,k,:]))
		lnp=lnp+np.log(px)
	return lnp

def plot_scatter(X, K, phi):
	pred=np.argmax(phi,1)
	fig=plt.figure()
	fig.suptitle('Problem 1.c K=%s' % (K))
	#plt.scatter([i[0] for i in X], [i[1] for i in X])
	colors=['r', 'g', 'b', 'purple', 'orange', 'black', 'yellow', 'grey']
	for k in range(K):
		plt.scatter([j[0] for j in X[np.where(pred==k)]], [j[1] for j in X[np.where(pred==k)]], color=colors[k])
	fig.savefig('Problem 1c K=%s.png' % (K))

def plot_lnp(lnp, K):
	fig=plt.figure()
	fig.suptitle('Problem 1.b for K=%s' %K)
	plt.plot(lnp)
	fig.savefig('Problem 1b for K=%s.png' %K)

def main():
	for k in np.arange(2,10,2):
		print k
		phi,n,mu,Lamb,pi,lnp,X=GMM(k)
		plot_scatter(X,k,phi)
		plot_lnp(lnp,k)


if __name__ == '__main__':
	main()	


