from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# Load dataset and skip the header row
dataset = loadtxt('pima_diabetes.csv', delimiter=',', skiprows=1)

# Split into input (X) and output (Y)
x = dataset[:, 0:8]
y = dataset[:, 8]

print("Input", x)
print("Output", y)

# Define the model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(x, y, epochs=10, batch_size=10)

# Evaluate the model
_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy * 100))

# Save the model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Save the weights
model.save_weights("weights.weights.h5")  
print("Model weights saved to disk.")

