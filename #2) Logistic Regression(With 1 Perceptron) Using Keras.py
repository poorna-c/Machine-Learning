import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def plot_decision_boundary(X,y,model):
    x_span = np.linspace(X[:,0].min(),X[:,0].max())
    y_span = np.linspace(X[:,1].min(),X[:,1].max())
    xx, yy = np.meshgrid(x_span,y_span)
    xx_, yy_ = xx.ravel(),yy.ravel()
    grid = np.c_[xx_,yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx,yy,z)

n_pts = 1000
Xa=np.array([np.random.normal(8,3,n_pts), np.random.normal(18,3,n_pts)]).T
Xb= np.array([np.random.normal(3,3, n_pts), np.random.normal(6,3, n_pts)]).T
X = np.vstack((Xa,Xb))
y = np.matrix(np.append(np.zeros(n_pts),np.ones(n_pts))).T

model = Sequential()
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer = Adam(lr = 0.06),loss='binary_crossentropy',metrics=['acc'])
h = model.fit(X,y,batch_size = 50, epochs=50,shuffle='true')
plt.plot(h.history['acc'],label = 'accuracy')
plt.legend()
plt.tight_layout()
plt.show()
plt.plot(h.history['loss'],label = "Loss")

plt.legend()
plt.tight_layout()
plt.show()

plot_decision_boundary(X,y,model)
plt.legend()
plt.tight_layout()
plt.show()

plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

plt.show()
plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])
x = 7.5
y = 5

point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="red")
print("prediction is: ",prediction)
plt.show()