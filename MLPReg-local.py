import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
import csv
import numpy
from sklearn.externals import joblib



x = np.arange(0.0, 5, 0.001).reshape(-1, 1)
y = np.sin(2 * np.pi * np.tan(x).ravel())

reg = MLPRegressor(hidden_layer_sizes=(1000,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

reg = reg.fit(x, y)

test_x = np.arange(0.0, 5, 0.05).reshape(-1, 1)
test_y = reg.predict(test_x)
print()
joblib.dump(reg, 'mlpreg.pkl')


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(x, y, s=1, c='b', marker="s", label='real')
ax1.scatter(test_x,test_y, s=20, c='r', marker="o", label='NN Prediction')
plt.show()