"""
This documents contains the dictionary survey_hyperparameters which specifies the
values to perform grid search over when training the mental health model.
Feel free to modify however you like, but be aware that adding too
many configurations will slow training down.
"""

survey_hyperparameters = {
    'input_dim': [8], # Do not change
    'output_dim': [2], # Do not change
    'hidden_dim': [1,5,10],
    'layers': [1,2,3],
    'learning_rate': [0.05,0.1,0.2],
    'epochs': [2,5],
    'batch_size': [2,5,10]
}