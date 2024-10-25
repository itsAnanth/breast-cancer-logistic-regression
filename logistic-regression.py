import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from data import getData


X_train, X_test, y_train, y_test, features, labels = getData()


class LogisticRegression:
    
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        return z * (1 - z)
    
    def binary_cross_entropy(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
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
    
    def fit(self, X: np.ndarray, y: np.ndarray, dump_weights=False, epoch_split=100, use_weights=False):
        
        if (use_weights):
            if (self._load_parameters()):
                print("Model parameters found, skipping training")
                return
            else:
                print("params.pkl does not exist, proceeding to training the model")
                        
        num_samples, num_features = X.shape
        
        self.weights = np.zeros(shape=(num_features, 1))
        self.bias = 0
        y = y.reshape(-1, 1)
        for epoch in range(self.epochs):
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            error = y_pred - y
            cost = self.binary_cross_entropy(y, y_pred)
            
            if (epoch % epoch_split == 0):
                print(f"{epoch}/{self.epochs} | loss = {cost}")
            
            dw = (1 / num_samples) * (X.T @ error)
            db = (1 / num_samples) * np.sum(error)
            
            self.weights += -self.learning_rate * dw
            self.bias += -self.learning_rate * db
            
        if not dump_weights:
            return
        
        with open('params.pkl', 'wb') as f:
            print("Dumping model parameters to params.pkl")
            params = {
                'weights': self.weights,
                'bias': self.bias
            }
            pk.dump(params, f)
            f.close()
        
    def predict(self, X, threshold=0.5):
        z = X @ self.weights + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred > threshold).astype(int)
    
    def accuracy(self, X, y):
        predictions: np.ndarray = self.predict(X)
        
        return np.mean(predictions.flatten() == y)
            
            
            
lr = LogisticRegression(epochs=10000)

lr.fit(X_train, y_train, dump_weights=True, use_weights=True)

print(f"test accuracy: {lr.accuracy(X_test, y_test) * 100}%")
print(f"train accuracy: {lr.accuracy(X_train, y_train) * 100}%")

        
