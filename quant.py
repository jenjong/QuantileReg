# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:48:54 2018

@author: Jeon
"""

import numpy as np
# simulation code
x = np.random.uniform(0,1,10)
ep = np.random.normal(0, np.exp(np.sin(2*np.pi*x)), 10)
y = np.sin(np.pi*x) + ep

# linear programming
# power decaying correlation matrix 1
p = 2
rho = 0.5
mu_vec = np.zeros(shape = p, dtype = "float32")
cov_mat = np.zeros(shape = (p,p), dtype= "float32")
for i in range(0,p):
    for j in range(0,p):
        cov_mat[i,j] = pow(rho, abs(i-j))
np.random.seed(1)        
z = np.random.multivariate_normal(mu_vec, cov_mat, 1000, 'ignore')
np.cov(z, rowvar = False)

# see https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
# see numpy.delete

np.repeat(1,3)


# error
        
cov_mat = [[0 for x in range(p)] for y in range(p)] 
for i in range(0,p):
    for j in range(0,p):
        cov_mat[i][j] = pow(rho, abs(i-j))
mu_vec = np.zeros(shape = p, dtype = "float32")
z = np.random.multivariate_normal(mu_vec, cov_mat, 1000).T

#np.linalg.det(cov_mat)
np.cov(z)


