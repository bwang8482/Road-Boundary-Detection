# Create first network with Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
name = 'train1'
dataset = numpy.loadtxt(name + '.out', delimiter=',')
# split into input (X) and output (Y) variables
X = dataset[:,0:64].astype(float)
Y = dataset[:,64].astype(int)
Y = (Y+1)/2

# create model
model = Sequential()
model.add(Dense(256, input_dim=64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=100,  verbose=2)
# calculate predictions
# predictions = model.predict(X)
# round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)

model_json = model.to_json()
with open(name + '.json', 'w') as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(name + '.h5')
print('Saved model to disk')