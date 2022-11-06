import itertools

from models import ClassificationModel


class GridSearch:
    def __init__(self, hyperparam_grid):
        self.hyperparam_grid = hyperparam_grid

    def train_and_eval(self, config, train_dataset, eval_dataset):
        """
        Returns a trained model based on the hyperparameters and its corresponding validation score

        Using the provided config, you should instantiate a ClassificationModel (the same you implemented earlier)
        train the model using ClassificationModel.train with the training dataset and the right elements from the config
        evaluate the model using ClassificationModel.eval

        :param config: A hyperparameter configuration dictionary specifying the value for each hyperparam
        :param train_dataset: A dataset to train on
        :param train_dataset: A dataset to evaluate on
        :return: model, score
        """
        model = ClassificationModel(input_dim=config['input_dim'], hidden_dim=config['hidden_dim'],
                                    layers=config['layers'], output_dim=config['output_dim'])
        model.train(train_dataset, learning_rate=config['learning_rate'], epochs=config['epochs'],
                    batch_size=config['batch_size'])
        score = model.eval(eval_dataset)
        return model, score

    def grid_search_configurations(self):
        """
        Returns a list of configuration dictionaries where each configuration dict has a single value for each
        hyperparameter from self.hyperparam_grid:
        {
            'learning_rate': 0.1,
            'batch_size': 32,
            'epochs': 10
        }

        Reminder: self.hyperparam_grid looks like:
        {
            'learning_rate': [0.1, 0.01],
            'batch_size': [32, 64],
            'epochs': [10,20]
        }

        #HINT: the function itertools.product might be helpful...
        combinations = list(itertools.product([0,1], [10,20], [100]))
        # OR
        values = [[0,1], [10,20], [100]]
        combinations = list(itertools.product(*values))
        assert combinations == [(0, 10, 100), (0, 20, 100), (1, 10, 100), (1, 20, 100)]

        """
        # Get hyperparameter names
        hp_names = sorted(self.hyperparam_grid.keys())

        # Create a list of lists of possible values for each hyperparameter
        values = [self.hyperparam_grid[hp] for hp in hp_names]

        # Use itertools.product to get a list of all combinations of hyperparams
        hp_configs = list(itertools.product(*values))

        # Using hp_configs and hp_names turn the list of value_combos
        configs_list = []
        for configValues in hp_configs:
            config = {}
            for i in range(len(hp_names)):
                config[hp_names[i]] = configValues[i]
            configs_list.append(config)

        return configs_list

    def perform_grid_search(self, train_dataset, eval_dataset):
        """This method should perform grid search over the hyperparameters and return the model with the best evaluation
        It should evaluate each model using the function

        It should use self.grid_search_configurations to get the configurations to run

        It should use self.train_and_eval to train and evaluate each configuration

        Assume self.hyperparam_grid includes keys:
            'input_dim', 'output_dim', 'hidden_dim', 'layers', 'learning_rate', 'batch_size', 'epochs'

        input_dim, hidden_dim, layers, and output_dim should be used to instantiate the model,
        the other hyperparameters should be used when running the model training.

        :returns best_model, best_eval, best_config:
            the model that performed best,
            the validation score it received
            the config parameters that gave it that model
        """
        best_model = None
        best_eval = float('-inf')
        best_config = None
        
        configsList = self.grid_search_configurations()
        for config in configsList:
            model, score = self.train_and_eval(config, train_dataset, eval_dataset)
            if score > best_eval:
                best_model = model
                best_eval = score
                best_config = config

        return best_model, best_eval, best_config
