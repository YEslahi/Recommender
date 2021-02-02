import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import logging
import pandas as pd
import data_handler

# TODO: use Argparse
# TODO: use sparse data structure
# TODO: write test
# TODO: visualize early
PATH = "../../data/ml-1m/1/"

# dataset_super = data_handler.SuperDataSet('ml1m', )
dataset_ml1m = data_handler.ml1m()

import code; code.interact(local=dict(globals(), **locals()))

input_data = data_loader(PATH)
input_shape = input_data['train_feature'].shape[0]
input_dim = input_data['train_feature'].shape[1]

# Create the model
model = Sequential()
model.add(Dense(16, input_shape=(input_shape,input_dim), activation='relu'))
model.add(Dense(input_dim, activation='sigmoid'))

# configure model and start training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(input_data['train_feature'], input_data['train_target'], epochs=2, batch_size=250, verbose=1, validation_split=0.2)

# test model 
test_results = model.evaluate(input_data['test_feature'], input_data['test_target'], verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

print("End!")