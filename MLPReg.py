import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.neural_network import MLPRegressor
import csv
import numpy
from sklearn.externals import joblib



x = np.arange(0.0, 1, 0.001).reshape(-1, 1)
y = np.sin(2 * np.pi * np.tan(x).ravel())

reg = MLPRegressor(hidden_layer_sizes=(500,),  activation='relu', solver='adam',    alpha=0.001,batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
               random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

reg = reg.fit(x, y)

#test_x = np.arange(0.0, 1, 0.05).reshape(-1, 1)
#test_y = reg.predict(test_x)
joblib.dump(reg, 'mlpreg.pkl')


