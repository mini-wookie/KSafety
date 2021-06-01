import warnings
import pandas as pd
import tensorflow as tf
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

path = 'https://raw.githubusercontent.com/mini-wookie/SafetyReport/main/ksafety.csv'
ksafety = pd.read_csv(path)
print(ksafety.columns)
ksafety.head()

x = ksafety[['traffic', 'fire', 'crime', 'suicide', 'virus', 'env', 'sexual', 'health', 'atmos', 'cctv', 'police', 'firefight', 'evac']]
y = ksafety[['safety']]
print(x.shape, y.shape)

X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X, Y)
model.compile(loss='mse')

model.fit(x, y, epochs=10000, verbose=1, batch_size=30)
model.fit(x, y, epochs=10, verbose=1, batch_size=30)

print(model.predict(x[0:5]))
print(y[0:5])

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=1)

# create linear regression object
reg = linear_model.LinearRegression()

# train the model using the training sets
reg.fit(X_train, y_train)

# regression coefficients
print('Coefficients: ', reg.coef_)

## variance score: 1 means perfect prediction
print('Variance score: {}'.format(reg.score(X_test, y_test)))

# plot for residual error

## setting plot style
plt.style.use('fivethirtyeight')

## plotting residual errors in training data
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train,
			color = "green", s = 10, label = 'Train data')

## plotting residual errors in test data
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test,
			color = "blue", s = 10, label = 'Test data')

## plotting line for zero residual error
plt.hlines(y = 0, xmin = 0, xmax = 50, linewidth = 3)

## plotting legend
plt.legend(loc = 'lower right')

## plot title
plt.title("KSafety")

## method call for showing the plot
plt.show()