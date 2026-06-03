from commands import *  

X = np.array([[1], [2], [3], [4]], dtype=float)

y = np.array([[2], [4], [6], [8]], dtype=float)

model = Sequential([
    Dense(1, input_shape=(1,))
])

model.compile(optimizer='sgd', loss='mse')

model.fit(X, y, epochs=100, verbose=0)

result = model.predict(np.array([[5.0]]), verbose=0)

print("Prediction for 5:", result[0][0])