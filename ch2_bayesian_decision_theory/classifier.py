import numpy as np
import math
from  matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def sample_gen(mu,sig,num_of_sample):
    '''
        :param mu: Mean vector
        :param sig: Covariance Matrix
        :param num_of_sample: Number of samples to be generated
        :return: Samples generated (Each column vector of matrix represents each sample data)
    '''
    return np.random.multivariate_normal(mu,sig,num_of_sample).T

def pdf_normal(vec,mu,sig):
    '''
        :param vec: Input data vector
        :param mu: mean vector
        :param sig: covariance matrix
        :return: probability density value
    '''
    det=np.linalg.det(sig)
    dim, = vec.shape
    sig_inv = np.linalg.inv(sig)
    f=(1.0/(math.sqrt(2*math.pi)*math.pow(det,dim/2.0)))*math.exp((-1/2)*np.matmul((vec-mu).T,np.matmul(sig_inv,(vec-mu))))
    return f

def norm_density_dis_fn(vec,mu,sig,prior_prob):
    det=np.linalg.det(sig)
    sig_inv=np.linalg.inv(sig)
    dim,=vec.shape
    f=(-1.0/2)*np.matmul((vec-mu).T,np.matmul(sig_inv,(vec-mu)))-(dim/2.0)*math.log(2.0*math.pi,math.e)-(1.0/2.0)*math.log(det,math.e)+math.log(prior_prob,math.e)
    return f
if __name__== '__main__':
    mu=[0,0]
    sig=[[1,0],[0,1]]
    z=sample_gen(mu,sig,1000)
    x,y=z[0],z[1]
    prob1=[]
    prob2=[]
    for val in z.T:
        prob1.append(norm_density_dis_fn(val,mu,sig,0.98))
        prob2.append(pdf_normal(val, mu, sig))


    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax1.plot(x, y,prob1,'go')
    ax1.plot(x, y, 'rv')
    ax2.plot(x, y, prob2, 'bo')
    ax2.plot(x,y,'rv')
    plt.show()
