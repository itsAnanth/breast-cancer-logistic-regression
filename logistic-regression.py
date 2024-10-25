import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from data import getData


"""
    data is obtained from sklearn load_breast_cancer() module
    data is normalized using sklearn StandardScaler normalizer
"""
X_train, X_test, y_train, y_test, features, labels = getData()


class LogisticRegression:
    
    def __init__(self, learning_rate=0.1, epochs=1000):
        
        # setting hyperparameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    # activation function used to introduce non linearity in the model
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    """
        derivative of sigmoid to be used in gradient calculation
        this is not explicity used in the code whatsoever since the derivative of sigmoid cancels out neatly
        with the derivative of binary cross entropy cost function
    """
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    """
        binary cross entropy cost function
        np.mean for the cost across entire samples
    """
    def binary_cross_entropy(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    """
        utility function to load weights instead of training the model from scratch
        
        note: the model must be trained atleast once to make use of this function
    """
    def _load_parameters(self):
        try:
            print("Loading model parameters from params.pkl")
            with open('params.pkl', 'rb') as f:
                params = pk.load(f)
                self.weights = params['weights']
                self.bias = params['bias']
                f.close()
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return False
    
    """
        actual training process
    """
    def fit(self, X: np.ndarray, y: np.ndarray, dump_weights=False, epoch_split=100, use_weights=False):
        
        # if model has been trained once, use pretrained weights
        if (use_weights):
            if (self._load_parameters()):
                print("Model parameters found, skipping training")
                return
            else:
                print("params.pkl does not exist, proceeding to training the model")
                        
        # training atleast once
        num_samples, num_features = X.shape
        
        self.weights = np.zeros(shape=(num_features, 1))
        self.bias = 0
        
        # convert labels to column vector to prevent shape mismatch
        y = y.reshape(-1, 1)
        for epoch in range(self.epochs):
            
            # pre activation
            z = X @ self.weights + self.bias
            # activated outputs
            y_pred = self.sigmoid(z)
            
            # error delta
            error = y_pred - y
            
            
            
            if (epoch % epoch_split == 0):
                # calculate cost at certain epoch split
                cost = self.binary_cross_entropy(y, y_pred)
                print(f"{epoch}/{self.epochs} | loss = {cost}")
            
            # gradient calculation and parameter updation
            dw = (1 / num_samples) * (X.T @ error)
            db = (1 / num_samples) * np.sum(error)
            
            self.weights += -self.learning_rate * dw
            self.bias += -self.learning_rate * db
            
            
        # skip writing parameters to file if user deems so
        if not dump_weights:
            return
        
        # else write to file
        with open('params.pkl', 'wb') as f:
            print("Dumping model parameters to params.pkl")
            params = {
                'weights': self.weights,
                'bias': self.bias
            }
            pk.dump(params, f)
            f.close()
     
    """
        prediction function
        predicted values are in a range 0 to 1 (Since sigmoid activation is used)
        map this prediction into binary classes based on a threshold value
        return the mapped values in integer format
        1 - malignant
        0 - benign
    """   
    def predict(self, X, threshold=0.5):
        z = X @ self.weights + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred > threshold).astype(int)
    
    """
        accuracy of model with a given input and label
        
        Note: this is not f1 score
    """
    def accuracy(self, X, y):
        predictions: np.ndarray = self.predict(X)
        
        return np.mean(predictions.flatten() == y)
            
            
            
lr = LogisticRegression(epochs=10000)

lr.fit(X_train, y_train, dump_weights=True, use_weights=True)

print(f"test accuracy: {lr.accuracy(X_test, y_test) * 100}%")
print(f"train accuracy: {lr.accuracy(X_train, y_train) * 100}%")

        
