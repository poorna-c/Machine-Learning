import matplotlib.pyplot as plt
import numpy as np

def draw(x1,x2):
    ln=plt.plot(x1,x2,'b')
    plt.pause(0.00001)
    ln[0].remove()

def sigmoid(score):
    return 1/(1+np.exp(-score))

def calculate_error(line_parameters, points , y):
    n=points.shape[0]
    p= sigmoid(points*line_parameters)
    cross_entropy=-(1/n)*(np.log(p).T*y + np.log(1-p).T*(1-y))
    return cross_entropy

def gradient_descent(line_parameters, points, y , alpha):
    n=points.shape[0]
    for i in range(100):
        p=sigmoid(points*line_parameters)
        gradient= points.T*(p-y)*(alpha/n)
        line_parameters = line_parameters - gradient

        w1=line_parameters.item(0)
        w2=line_parameters.item(1)
        b=line_parameters.item(2)

        x1=np.array([points[:,0].min(), points[:,0].max()])
        x2= -b/w2 + (x1*(-w1/w2))
        draw(x1,x2) 
    return x1,x2
  
n_pts = 600
bias= np.ones(n_pts)
top_region=np.array([np.random.normal(8,3,n_pts), np.random.normal(18,3,n_pts), bias]).T
bottom_region= np.array([np.random.normal(3,3, n_pts), np.random.normal(6,3, n_pts), bias]).T
all_points=np.vstack((top_region, bottom_region))

line_parameters = np.matrix([1,0.1,0.1]).T

y=np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts*2, 1)
_, ax= plt.subplots(figsize=(6,6))
ax.scatter(top_region[:,0], top_region[:,1], color='r')
ax.scatter(bottom_region[:,0], bottom_region[:,1], color='y')
plt.tight_layout()
x1,x2 = gradient_descent(line_parameters, all_points, y , 0.08)
plt.show()