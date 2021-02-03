import logging
import pandas as pd
import data_handler
import model_handler
import argparse

# TODO: use sparse data structure
# TODO: write test
# TODO: visualize early
# TODO: report all the timing numbers

# add arguments
parser = argparse.ArgumentParser(description='MLP')
parser.add_argument('--train_epoch', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=1500)
parser.add_argument('--learning_rates', type=float,
                    choices=[0.0001, 0.001, 0.002, 0.003], default=0.003)
parser.add_argument('--optimizer_method', choices=['Adam', 'Adadelta', 'Adagrad', 'RMSProp', 'GradientDescent',
                                                   'Momentum'], default='Adam')
parser.add_argument(
    '--activation', choices=['sigmoid', 'relu', 'Elu', 'Tanh', "Identity"], default='sigmoid')
parser.add_argument('--metrics', choices=['accuracy'])

args = parser.parse_args()

dataset_ml1m = data_handler.Ml1m()
input_data = dataset_ml1m.load_dataset()

model_obj = model_handler.MLP(args, input_data)
model_obj.instantiate()
model_obj.compile()
model_obj.fit()
test_results = model_obj.evaluate()

print("End!")