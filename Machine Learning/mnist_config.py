"""
This documents contains the configuration dictionary for MNIST digit classification

You set the hyperparameter values when training.
"""
student_config = {
    'input_dim': 784,  # DO NOT CHANGE
    'output_dim': 10,  # DO NOT CHANGE
    # Change the following
    'hidden_dim': 200,
    'layers': 4,
    'learning_rate': 0.05,
    'epochs': 4,
    'batch_size': 10
}
