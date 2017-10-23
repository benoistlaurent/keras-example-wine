
import pandas
from keras.models import Sequential
from keras.layers import Dense


# Read input data
df = pandas.read_csv('datasets/wine.csv')


# Create network input X and output Y.
# X is made by removing the label column.
# Y is made of the label column. 
X = df.drop(['white'], axis=1).values
Y = df['white']


# Just for fun: take the first 10 values to use it later for prediction (and remove it from train)
X_pred = X[:10,]
Y_pred = Y[:10]

X = X[10:,]
Y = Y[10:]


# Number of input variables.
nvars = X.shape[1]


# Create model
model = Sequential()
model.add(Dense(30, input_dim=nvars, activation='relu'))
model.add(Dense(160, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
model.fit(X, Y, epochs=20, batch_size=10)


# Evaluate the model.
scores = model.evaluate(X, Y)
print("\n{}: {:.2f}%".format(model.metrics_names[1], scores[1] * 100))


# Calculate predictions
predictions = model.predict(X_pred)
print(predictions)

