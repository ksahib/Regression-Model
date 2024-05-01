from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
# pylint: disable=import-error
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
# pylint: disable=import-error
from tensorflow.keras.callbacks import TensorBoard

data = pd.read_csv("car_purchasing.csv")
data.head()

X_train, X_test, y_train, y_test = train_test_split(
    data[['age', 'annual Salary', 'credit card debt', 'net worth']], data.amount, test_size=0.2, random_state=42)
print(X_train)

model = tf.keras.Sequential([
    Dense(4, input_shape=(4, )),
    Dense(64, activation='relu'),
    Dense(1, activation='linear')

])
tc = TensorBoard(log_dir='./logs', histogram_freq=1)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

model.fit(X_train, y_train, epochs=5000, batch_size=30, validation_split=0.2, callbacks=[tc])
loss = model.evaluate(X_test, y_test, batch_size=30, verbose=1)
print(loss)

y_pred = model.predict(X_test)
print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.flatten()}).head())
plt.scatter(y_test, y_pred, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
