# digit-classifier-neural-network
A simple neural network implementation to classify the MNIST handwritten digit set, implemented without the use of pre-existing machine learning libraries.


## Data Description
Training and testing data are .csv files representing 28x28 grayscale images of handwritten digits. Each of the 784 pixels of these images is represented by a single 8-bit color channel in a column of the input .csv file.

Training labels are the associated integer digits between 0 and 9

Training data and labels and testing data .csv files may be specified on the command line as follows:
```
python3 nn.py <training_data> <training_labels> <testing_data>
```
If no command line arguments are provided, the program will use existing data and labels in the current directory.


## Network Implementation
The network is a multi-layer perceptron network with a 784-neuron input layer, two hidden layers of 128 neurons each, and a 10-neuron output layer (1 for each possible digit). 

Each hidden layer uses a sigmoid activation function, and the output layer uses a softmax activation function in order to do multi-class classification. Loss is computed using cross-entropy loss and backprogpogated through the network.

Learning rate, batch size, number of epochs, and number of neurons in each hidden layer may be adjusted as needed in the network constructor. 

Testing label predictions are integer digits output into a .csv file `test_predictions.csv` for external comparison with true values (if available). Testing accuracy based on default network hyperparameter values is 93-95% for testing set sizes between 10,000-60,000 and training set sizes between 10,000-60,000.
