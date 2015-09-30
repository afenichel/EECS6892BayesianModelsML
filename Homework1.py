import scipy
from scipy import io, stats
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('mnist_mat.mat')


def prior(y):
	e=1
	f=1
	prior={}
	N=mat['ytest'].size
	if y==1:
		prior=float(e+np.where(mat['ytrain']==y)[0].size)/float(N+e+f)
	if y==0:
		prior=float(f+np.where(mat['ytrain']==y)[0].size)/float(N+e+f)
	return prior

def likelihood(y, i):
	a=1
	b=1
	c=1
	x_new=mat['Xtest'][:,i]
	x=mat['Xtrain'][:,np.where(mat['ytrain']==y)[1]]
	n=x.shape[1]
	v=2*b+n
	scale=2*float((a*n+a+1))/float((v+a*n*v))
	mean=np.mean(x,1)
	mu=n*mean/float((1/float(a)+n))
	t=sum((-(v+1)/2)*np.log(1+np.power(np.subtract(x_new,mu),2)/(scale*v)))
	return t

def posterior(y, i):
	l=likelihood(y, i)
	p=prior(y)
	pr_y=l*p
	return pr_y

def predictions():
	r=mat['ytest'].size
	y_guess=np.zeros(r)
	for i in range(r):
		pr_0=posterior(0, i)
		pr_1=posterior(1, i)
		y_hat=max(pr_0, pr_1)
		if y_hat==pr_0:
			y_guess[i]=0
		if y_hat==pr_1:
			y_guess[i]=1
	return y_guess

def confusion_matrix():
	y_guess=predictions()
	idx4=np.where(mat['ytest'][0]==0)
	idx9=np.where(mat['ytest'][0]==1)
	misclassified_4_idx=np.where(y_guess[idx4]==1)[0]
	classified_4_idx=np.where(y_guess[idx4]==0)[0]
	misclassified_9_idx=np.where(y_guess[idx9]==0)[0]
	classified_9_idx=np.where(y_guess[idx9]==1)[0]
	number_4_classified=classified_4_idx.size
	number_4_misclassified=misclassified_4_idx.size
	number_9_classified=classified_9_idx.size
	number_9_misclassified=misclassified_9_idx.size
	c_matrix=pd.DataFrame(columns=['Classified as 4', 'Classified as 9'], index=['4', '9'])
	c_matrix['Classified as 4']['4']=number_4_classified
	c_matrix['Classified as 9']['4']=number_4_misclassified
	c_matrix['Classified as 9']['9']=number_9_classified
	c_matrix['Classified as 4']['9']=number_9_misclassified
	return c_matrix

def misclassified_digits(seed):
	probability=prediction_probability()
	np.random.seed(seed)
	y_guess=predictions()
	misclassified=mat['ytest'][0]-y_guess
	m=np.where(misclassified!=0)[0]
	r=np.random.choice(m,3)
	image_prediction(r, y_guess, probability)

def ambiguous_predictions():
	probability=prediction_probability()
	y_guess=predictions()
	a=np.power(0.5-probability[0], 2)
	idx_min=a.argsort()[0:3]
	image_prediction(idx_min, y_guess, probability)

def prediction_probability():
	r=mat['ytest'].size
	pr_0=np.zeros(r)
	pr_1=np.zeros(r)
	for i in range(r):
		pr_0[i]=posterior(0, i)
		pr_1[i]=posterior(1, i)
	pr=np.vstack((pr_0, pr_1))
	probability=1-pr/np.vstack((np.sum(pr,0),np.sum(pr,0)))
	return probability

def image_prediction(index_array, y_guess, probability):	
	p={}
	fig = plt.figure(figsize=(15,5))
	for i in index_array:
		idx=list(index_array).index(i)
		binary_map=np.asarray([4,9])
		y_actual=mat['ytest'][0][i]	
		classification=binary_map[y_guess[i]]
		actual=binary_map[y_actual]
		img=np.reshape(np.dot(mat['Q'], mat['Xtest'])[:,i], (28,28))
		title='Pr(y*= %s |x*, Xtrain, ytrain) = %s \n Bayes Classifier guess = %s \n ytest = %s' %(actual, '{:.4%}'.format(probability[y_actual][i]), classification, actual)
		p[idx]=fig.add_subplot(1,3,idx+1)
		p[idx].set_title(title, fontsize=10)
		p[idx].axis('off')
		p[idx].imshow(img)
	fig.show()	
	
	
