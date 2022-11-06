import numpy as np

import nn


class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.w)


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        return 1 if nn.as_scalar(self.run(x)) >= 0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        notConverge = True
        while notConverge:
            notConverge = False
            for x, y in dataset.iterate_once(1):
                predictionScalar = self.get_prediction(x)
                desireScalar = nn.as_scalar(y) 
                match = (desireScalar == predictionScalar)
                if not match:
                    self.get_weights().update(nn.Constant(desireScalar*x.data), 1)
                notConverge |= (not match)
                



class LogisticRegressionModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Logistic Regression instance.

        A perceptron classifies data points as either belonging to a particular
        class (1) or not (0). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)


    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the model.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the logistic regression to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the predicted probability)
        """
        return nn.Sigmoid(nn.DotProduct(x, self.w))

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or 0
        """
        return 1 if nn.as_scalar(self.run(x)) > 0.5 else 0

    def calculate_log_likelihood_gradient(self, x, y):
        """
        Calculate the maximum likelihood gradient for a single datapoint x and y

        nn.Subtract and nn.ScalarMatrixMultiply might be helpful to use
        Returns: a Node representing the gradient of the maximum log-likelihood of the data with respect to the params
        """
        # raise NotImplementedError()
        return nn.Constant(nn.Subtract(y, self.run(x)).data * x.data)

    def train(self, dataset, learning_rate, iterations):
        """
        Train the logistic regression model using stochastic gradient ascent (a single datapoint at a time).

        Use dataset.pick_random(batch_size) to sample random (x, y) data pairs from the dataset

        Use the update function on your parameter to make further changes.
        """
        for i in range(iterations):
            x,y = dataset.pick_random(1)
            likelihoodGradient = self.calculate_log_likelihood_gradient(x, y)
            self.w = nn.Add(self.w, nn.Constant(learning_rate*likelihoodGradient.data))


class ClassificationModel(object):
    """
    A generic neural network model

    The model should be initialized using the provided input hyperparameters.
    """
    def __init__(self, input_dim=784, hidden_dim=4, layers=2, output_dim=10):
        """Initialize your model parameters here"""
        self.weights = []
        self.biases = []

        if layers < 1:
            raise ValueError('Too few layers')

        if layers < 2:
            self.weights.append(nn.Parameter(input_dim, output_dim))
            self.biases.append(nn.Parameter(1, output_dim))
        else:
            self.weights.append(nn.Parameter(input_dim, hidden_dim))
            for i in range(layers - 2):
                self.weights.append(nn.Parameter(hidden_dim, hidden_dim))
            self.weights.append(nn.Parameter(hidden_dim, output_dim))

            for i in range(layers - 1):
                self.biases.append(nn.Parameter(1, hidden_dim))
            self.biases.append(nn.Parameter(1, output_dim))

        self.params = self.weights + self.biases

    def get_params(self):
        """Should return a list of all the parameters"""
        return self.params

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """

        layerResults = x
        for i in range(len(self.weights) - 1):
            layerWeights = self.weights[i]
            layerBiases = self.biases[i]
            layerResults = nn.ReLU(nn.AddBias(nn.Linear(layerResults, layerWeights), layerBiases))

        lastLayerWeights = self.weights[len(self.weights) - 1]
        lastLayerBiases = self.biases[len(self.biases) - 1]

        return nn.AddBias(nn.Linear(layerResults, lastLayerWeights), lastLayerBiases)


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset, learning_rate, epochs, batch_size):
        """
        Trains the model.
        """
        for j in range(epochs):
            for x,y in dataset.iterate_once(batch_size):
                gradients = nn.gradients(self.get_loss(x,y), self.get_params())
                gradWeights = gradients[0:int(len(gradients)/2)]
                gradBiases = gradients[int(len(gradients)/2):]
                for i in range(len(self.weights)):
                    # print("weights before update:", self.weights[i].data)
                    self.weights[i].update(gradWeights[i], -learning_rate)
                    # print("weights after update:", self.weights[i].data)

                    # print("biases before update:", self.biases[i].data)
                    self.biases[i].update(gradBiases[i], -learning_rate)
                    # print("biases after update:", self.biases[i].data)

    def eval(self, eval_dataset):
        """
        Runs evaluation using the accuracy metric. You do not need to implement this part but should take a look.
        """
        pred_label = None
        true_label = None
        for x,y in eval_dataset.iterate_once(batch_size=len(eval_dataset)):
            pred_label = np.argmax(self.run(x).data, axis=1)
            true_label = np.argmax(y.data, axis=1)

        return (pred_label == true_label).mean()

    def get_prediction(self, x):
        """
        Gets a list of predictions for a databatch x
        """
        return np.argmax(self.run(x).data, axis=1)


