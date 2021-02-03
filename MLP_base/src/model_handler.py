import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from abc import ABC, abstractmethod  #abstract base class

class Model(ABC):
    """A class that creates model, fit, and evaluate it."""
    def __init__(self, args, input_data, model):
        self.args = args
        self.train_epoch = args.train_epoch
        self.batch_size = args.batch_size
        self.learning_rates = args.learning_rates
        self.optimizer_method = args.optimizer_method
        self.activatoin = args.activation
        self.metrics = args.metrics
        self.input_data = input_data
        self.input_shape = input_data['train_feature'].shape
        self.model = model

    @abstractmethod
    def instantiate(self):
        pass
    
    @abstractmethod
    def compile(self):
        pass

    def fit(self):
        self.model.fit(self.input_data['train_feature'], self.input_data['train_target'], epochs=self.train_epoch, batch_size=self.batch_size, verbose=1, validation_split=0.2)
    
    def evaluate(self):
        # test model 
        test_results = self.model.evaluate(self.input_data['test_feature'], self.input_data['test_target'], verbose=1)
        print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')
        return test_results

class MLP(Model):
    def __init__(self, args, input_data):
        self.model = Sequential()
        super().__init__(args, input_data, self.model)
    
    def instantiate(self):
        self.model.add(Dense(16, input_shape=self.input_shape, activation=self.activatoin))
        self.model.add(Dense(self.input_shape[1], activation=self.activatoin))

    def compile(self):
        # configure model and start training
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer_method, metrics=['accuracy'])