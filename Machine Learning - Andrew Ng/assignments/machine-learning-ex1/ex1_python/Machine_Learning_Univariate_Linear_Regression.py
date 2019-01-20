#!/usr/bin/env python
# coding: utf-8

# # Machine Learning using Univariate Linear Regression (Ex1)
# 
# Estimating the profit from a city basing on its population size.
# 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import meshgrid, cm
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter

sns.set(style="whitegrid")


# ## ======================= Part 2: Plotting =======================

# In[2]:


print('Plotting Data ...\n')
data = pd.read_csv('ex1data1.txt', names = ['population_size', 'profit'])
print(data.head())


# In[3]:


x = data['population_size'] # training set
y = data['profit'] # labels
m = len(y) # training size


# In[4]:


# plots the data
# f, ax = plt.subplots(figsize=(6.5, 6.5))
# sns.despine(f, left=True, bottom=True)
# sns.scatterplot(data=data, x="population_size", y="profit", sizes=(1, 8))
sns.jointplot('population_size', 'profit', data=data, color="m", height=7)


# ## =================== Part 3: Cost and Gradient descent ===================

# Add a column of ones to X to facilitate the manipulation.
# 
# Each row is a input with the following format:
# 
# $X[0] = [ x_0, x_1 ]$ where $x_0 = 1$

# In[5]:


# Add a column of ones to X to facilitate the manipulation
X = np.column_stack((np.ones(m), x))


# Initialize fitting parameters $\theta = [0, 1]^T$ (fixed to 0 to validated the output from computCost and gradient descent).
# 
# Learning rate $\alpha = 0.01$.

# In[6]:


theta = np.zeros([2])
# Some gradient descent settings
iterations = 1500;
alpha = 0.01; # learning rate


# In[7]:


print(np.shape(X))
print(X[:5])
print(np.shape(y))
print(y[:5])
print(np.shape(theta))
print(theta)


# ### Hypothesis Function
# Function that defines our linear model.
# 
# Definition:
# 
# $h_\theta(x) = \theta_0 + \theta_1 * x_1$
# 
# Vectorial form:
# 
# $h_\theta(x) = \theta^{T} * x$
# 
# where:
# $x = [x_0, x_1]$; $x_0 = 1$ and $\theta = [\theta_0, \theta_1]$

# In[8]:


def hypothesis(X, theta):
    # return [np.dot(xi, theta) for xi in X]
    return X.dot(theta)


# ### Compute cost for linear regression
# `computeCost` computes the cost of using theta as the parameter for linear regression to fit the data points in X and y.
# 
# Function cost:
# 
# $ J(\theta) = \frac{1}{2m} \sum_{i=0}^{m} (h_\theta(x^{(i)}) - y^{(i)})^{2} $
# 
# Expecting 32.07 at first iteration as $\theta$ was initialized with $[0, 0]$.

# In[9]:


"""
Inputs:
X = [
  [ x0, x1 ]
]
y = [
  [ ]
]
theta = [ theta_0, theta_1 ]
"""
def computeCost(X, y, theta):
    m = len(y)
    h_theta = hypothesis(X, theta)
    # return (1 / (2 * m)) * ((h_theta - y) ** 2).sum()
    delta = (h_theta - y)
    return (1 / (2 * m)) * delta.T.dot(delta)


# Testing the cost function with $\theta = [0, 0]^T$

# In[10]:


# compute and display initial cost
J = computeCost(X, y, [0, 0]);
print('Cost computed = %f' % J);
print('Expected cost value (approx) 32.07\n');


# Testing the cost function with $\theta = [-1, 2]^T$

# In[11]:


# further testing of the cost function
J = computeCost(X, y, [-1, 2]);
print('Cost computed = %f' % J);
print('Expected cost value (approx) 54.24\n');


# ### Running Gradient Descent
# We use gradient descent to find the parameters values $\theta$ that **minimize** $J$.
# In each iteration we calculate a $\theta'$ where $J(\theta') < J(\theta)$.
# 
# This $\Theta'$ defined by $\theta' = \theta - \alpha * \nabla h_\theta$, where $\nabla h_\theta$ is the amount we need change to *minimize* $J(\theta)$ and $\alpha$ is the step we will take.
# 
# If $\alpha$ is too large we could ending increasing $J$, it need to be small enough to converge quickly to the points the $J(\Theta)$ is minimum.
# 
# 
# Step to update each parameter:
# 
# $\theta_j := \theta_j - \alpha * \frac{\partial J}{\partial \theta_j} $
# 
# Where:
# 
# $\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} [( h_\theta(x^{(i)}) - y^{(i)})$ when $j = 0$ (bacause it is the bias - doesn't have a feature).
# 
# $\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} [( h_\theta(x^{(i)}) - y^{(i)}) * x^{(i)}]$ when $j = 1$.
# 
# Metrix form:
# 
# $ \frac{\partial J}{\partial \theta_j} = \frac{1}{m} = X^{T} ( h_\theta(x^{(i)}) - y^{(i)}) $
# 
# 
# `gradientDescent(X, y, theta, alpha, num_iters)` performs gradient descent to learn $\theta$ parameters.
# 
# It return the an array with $\theta$ containing the values found by taking num_iters gradient steps with learning rate alpha.
# 
# Also it return a array with the history of $J(\theta)$ to be plotted.
# 

# In[12]:


# calculate the theta_0 and theta_1 individually
def gradientDescentManualWay(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        h_theta = hypothesis(X, theta)
        # update the theta individually
        nabla_theta_0 = (1 / m) * (h_theta - y).sum() # theta 0
        nabla_theta_1 = (1 / m) * ((h_theta - y) * X.transpose()[1]).sum() # theta 1

        # theta[0] = theta[0] - alpha * nabla_theta_0
        # theta[1] = theta[1] - alpha * nabla_theta_1
        theta = theta - alpha * np.array([nabla_theta_0, nabla_theta_1])
        
        # Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta);
    return theta, J_history

# better way using the convenient x_0 = 1
def gradientDescentVectorialWay(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    
    for i in range(num_iters):
        h_theta = hypothesis(X, theta)
        nabla = (1 / m) * np.dot((h_theta - y).transpose(), X);
        
        theta = theta - alpha * nabla.transpose()
        
        # Save the cost J in every iteration
        J_history[i] = computeCost(X, y, theta);
    return theta, J_history


# In[13]:


print('\nRunning Gradient Descent ...\n')
theta = np.zeros(2)
theta, J_history = gradientDescentVectorialWay(X, y, theta, alpha, iterations);

# print theta to screen
print('Theta found by gradient descent:');
print(theta);
print('\nExpected theta values (approx)');
print('[ -3.6303,  1.1664]');


# In[14]:


print('Predict values for population sizes of 35,000 and 70,000')

predict1 = hypothesis(np.array([[1, 3.5]]), theta)
print('For population = 35,000, we predict a profit of %f\n' % (predict1[0] * 10000))

predict2 = hypothesis(np.array([[1, 7]]), theta)
print('For population = 70,000, we predict a profit of %f\n' % (predict2[0] * 10000))


# ## ============= Part 4: Visualizing $J(\theta_0, \theta_1)$ =============

# In[15]:


theta0_vals = np.arange(-10, 10, 2)
theta1_vals = np.arange(-1, 4, 0.5)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros([len(theta0_vals), len(theta1_vals)])

# Fill out J_vals
for t0 in range(len(theta0_vals)):
    for t1 in range(len(theta1_vals)):
        J_vals[t0][t1] = computeCost(X, y, [theta0_vals[t0], theta1_vals[t1]])

gX, gY = meshgrid(theta0_vals, theta1_vals)


# In[16]:


print('Visualizing J(theta_0, theta_1) ...')
fig = plt.figure()
ax = Axes3D(fig)

surf = ax.plot_surface(gX, gY, J_vals, rstride=1, cstride=1, cmap=cm.RdBu, linewidth=0, antialiased=False)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)


# In[17]:


# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
fig = plt.figure()
ax = Axes3D(fig)

cset = ax.contour(gX, gY, J_vals.transpose(), 16, extend3d=True)
ax.clabel(cset, fontsize=9, inline=1)
fig.colorbar(cset, shrink=0.5, aspect=5)


# ### ============= Extra: Visualizing Learning Curve =============

# In[18]:


# Plot the linear fit
sns.lineplot(range(iterations), J_history)


# ### ============= Extra: Plotting Hypothesis Model =============

# In[19]:


population_size = np.arange(0, 25, 2.5)
profit = np.zeros(len(population_size))

for i in range(len(population_size)):
    predict = hypothesis(np.array([[1, population_size[i]]]), theta)
    profit[i] = predict[0]

g = sns.FacetGrid(data, height=6)
g = g.map(plt.scatter, 'population_size', 'profit', edgecolor="w")
plt.plot(population_size, profit, color='r')


# In[ ]:




