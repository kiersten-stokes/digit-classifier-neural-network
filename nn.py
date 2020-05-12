import sys
import time
import numpy as np


DIGITS = 10
NUM_POSS_VALUES = 255.0

class NN:

    INPUT_SIZE = 784
    OUTPUT_SIZE = 10
    
    DECAY = 0.0016

    def __init__(self, data, labels, alpha=1.0, epochs=40, neurons=128, batch_size=10):
        
        self.X = data
        self.y = labels
        
        self.num_samples = data.shape[0]
        
        self.alpha = alpha
        self.epochs = epochs
        self.neurons = neurons
        self.batch_size = batch_size
        
        
        # initialize weights and biases for each layer
        self.w1 = np.random.randn(self.INPUT_SIZE, neurons)
        self.b1 = np.zeros((1, neurons))
        
        self.w2 = np.random.randn(neurons, neurons)
        self.b2 = np.zeros((1, neurons))
        
        self.w3 = np.random.randn(neurons, self.OUTPUT_SIZE)
        self.b3 = np.zeros((1, self.OUTPUT_SIZE))
        
        
    
    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))
    
    
    def sigmoid_prime(self, z):
        return z * (1.0 - z)
    
   
    def softmax(self, z):
        res = np.exp(z - np.max(z, axis=1, keepdims=True))
        return res / np.sum(res, axis=1, keepdims=True)
    
    
    def feed_forward(self, X):
        z1 = np.dot(X, self.w1) + self.b1
        self.a1 = self.sigmoid(z1)
        
        z2 = np.dot(self.a1, self.w2) + self.b2
        self.a2 = self.sigmoid(z2)
        
        z3 = np.dot(self.a2, self.w3) + self.b3
        self.a3 = self.softmax(z3)
    

    
    def cross_entropy(self, predicted, actual):
        res = predicted - actual
        num_samples = actual.shape[0]
        return res / num_samples
    
    
    def back_propogate(self, X, y):
        # get gradients
        a3_delta = self.cross_entropy(self.a3, y)
        
        z2_delta = np.dot(a3_delta, self.w3.T)
        a2_delta = z2_delta * self.sigmoid_prime(self.a2)
        
        z1_delta = np.dot(a2_delta, self.w2.T)
        a1_delta = z1_delta * self.sigmoid_prime(self.a1)
        
        self.w3 = self.w3 - self.alpha * np.dot(self.a2.T, a3_delta)
        self.b3 = self.b3 - self.alpha * np.sum(a3_delta, axis=0, keepdims=True)
        
        self.w2 = self.w2 - self.alpha * np.dot(self.a1.T, a2_delta)
        self.b2 = self.b2 - self.alpha * np.sum(a2_delta, axis=0)
        
        self.w1 = self.w1 - self.alpha * np.dot(X.T, a1_delta)
        self.b1 = self.b1 - self.alpha * np.sum(a1_delta, axis=0)

    
    def predict(self, data):
        self.feed_forward(data)
        return self.a3.argmax()

    
    def evaluate(self, x, y):
        predictions = [self.predict(i) for i in x]
        num_correct = sum(int(prediction == np.argmax(y)) for (prediction, y) in zip(predictions, y))
        num_samples = x.shape[0]
        
        return num_correct / num_samples * 100
        
    
    def train(self):
        
        # break data into batches
        X = np.array_split(self.X, self.num_samples/self.batch_size)
        y = np.array_split(self.y, self.num_samples/self.batch_size)
            
        for epoch in range(self.epochs):
            
            '''
            # adjust alpha based on epoch number and decay rate
            self.alpha = self.alpha * (1 / (1 + self.DECAY * epoch))
            '''
            print("epoch #: ", epoch)
            
            for X_batch, y_batch in zip(X, y):
                self.feed_forward(X_batch)
                self.back_propogate(X_batch, y_batch)
            
                
    
if __name__ == "__main__":  
    
    # suppress numpy exp overflow warning
    np.warnings.filterwarnings('ignore', '(overflow|invalid)')
    
    train_image_fp = 'train_image.csv'
    train_label_fp = 'train_label.csv'
    test_image_fp = 'test_image.csv'
    
    if len(sys.argv) > 1:
        train_image_fp, train_label_fp, test_image_fp = sys.argv[1:4]
    
    
    # read in training and test data and labels
    X_train = np.genfromtxt(train_image_fp, delimiter=",", dtype="float64")                  # size: (num_samples, 784)
    y_train = np.genfromtxt(train_label_fp, delimiter=",", dtype="uint64").reshape(-1)       # size: (num_samples,)
    X_test = np.genfromtxt(test_image_fp, delimiter=",", dtype="float64")                    # size: (num_samples, 784)
    
    
    # onehot-encode labels for softmax classification
    y_train_onehot = np.eye(DIGITS)[y_train]        # size: (num_samples, 10)
        
        
    # initialize neural network with unit normalized training values and labels, then train
    network = NN(X_train/NUM_POSS_VALUES, y_train_onehot)
    network.train()
    
    
    # write predictions on unit normalized test set to a .csv file
    labels = [network.predict(x) for x in X_test/NUM_POSS_VALUES]
    np.savetxt("test_predictions.csv", labels, delimiter=",", fmt="%i")
    