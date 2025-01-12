from numpy import loadtxt
from keras.models import model_from_json  # Fixed typo here

dataset = loadtxt('pima_diabetes.csv', delimiter=',', skiprows=1)

x = dataset[:, 0:8]
y = dataset[:, 8]

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights("weights.weights.h5")
print("Loaded model from the disk")

predictions = model.predict(x)

for i in range(25, 35):
    print('%s => %.2f (expected %.0f)' % (x[i].tolist(), predictions[i], y[i]))
