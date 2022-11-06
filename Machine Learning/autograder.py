# A custom autograder for this project

################################################################################
# A mini-framework for autograding
################################################################################

import optparse
import pickle
import random
import sys
import traceback

class WritableNull:
    def write(self, string):
        pass

    def flush(self):
        pass

class Tracker(object):
    def __init__(self, questions, maxes, prereqs, mute_output, leaderboard_name="leaderboard", vocareum_grade_file=None):
        self.questions = questions
        self.maxes = maxes
        self.prereqs = prereqs

        self.points = {q: 0 for q in self.questions}

        self.current_question = None

        self.current_test = None
        self.points_at_test_start = None
        self.possible_points_remaining = None

        self.mute_output = mute_output
        self.original_stdout = None
        self.muted = False

        self.leaderboard_name = leaderboard_name
        self.leaderboard_score = 0

        self.vocareum_grade_file = vocareum_grade_file

    def mute(self):
        if self.muted:
            return

        self.muted = True
        self.original_stdout = sys.stdout
        sys.stdout = WritableNull()

    def unmute(self):
        if not self.muted:
            return

        self.muted = False
        sys.stdout = self.original_stdout

    def begin_q(self, q):
        assert q in self.questions
        text = 'Question {}'.format(q)
        print('\n' + text)
        print('=' * len(text))

        for prereq in sorted(self.prereqs[q]):
            if self.points[prereq] < self.maxes[prereq]:
                print("""*** NOTE: Make sure to complete Question {} before working on Question {},
*** because Question {} builds upon your answer for Question {}.
""".format(prereq, q, q, prereq))
                return False

        self.current_question = q
        self.possible_points_remaining = self.maxes[q]
        return True

    def begin_test(self, test_name):
        self.current_test = test_name
        self.points_at_test_start = self.points[self.current_question]
        print("*** {}) {}".format(self.current_question, self.current_test))
        if self.mute_output:
            self.mute()

    def end_test(self, pts):
        if self.mute_output:
            self.unmute()
        self.possible_points_remaining -= pts
        if self.points[self.current_question] == self.points_at_test_start + pts:
            print("*** PASS: {}".format(self.current_test))
        elif self.points[self.current_question] == self.points_at_test_start:
            print("*** FAIL")

        self.current_test = None
        self.points_at_test_start = None

    def end_q(self):
        assert self.current_question is not None
        assert self.possible_points_remaining == 0
        print('\n### Question {}: {}/{} ###'.format(
            self.current_question,
            self.points[self.current_question],
            self.maxes[self.current_question]))

        self.current_question = None
        self.possible_points_remaining = None

    def finalize(self):
        import time
        print('\nFinished at %d:%02d:%02d' % time.localtime()[3:6])
        print("\nProvisional grades\n==================")

        for q in self.questions:
          print('Question %s: %s/%d' % (q, self.points[q], self.maxes[q]))
        print('------------------')
        print('Total: %s/%d' % (sum(self.points.values()),
            sum([self.maxes[q] for q in self.questions])))

        print("""Make sure to submit on vocareum!""")

        if self.vocareum_grade_file is not None:
            with open(self.vocareum_grade_file, "a") as f:
                for q in self.questions:
                    f.write("%s, %s\n" % (q, self.points[q]))
                f.write(f'{self.leaderboard_name}, {self.leaderboard_score}\n')
                f.write('Total, %s\n' % sum(self.points.values()))

    def add_points(self, pts):
        self.points[self.current_question] += pts

    def set_leaderboard(self, score):
        self.leaderboard_score = score

TESTS = []
PREREQS = {}
def add_prereq(q, pre):
    if isinstance(pre, str):
        pre = [pre]

    if q not in PREREQS:
        PREREQS[q] = set()
    PREREQS[q] |= set(pre)

def test(q, points):
    def deco(fn):
        TESTS.append((q, points, fn))
        return fn
    return deco

def parse_options(argv):
    parser = optparse.OptionParser(description = 'Run public tests on student code')
    parser.set_defaults(
        edx_output=False,
        gs_output=False,
        no_graphics=False,
        mute_output=False,
        check_dependencies=False,
        )
    parser.add_option('--edx-output',
                        dest = 'edx_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_option('--gradescope-output',
                        dest = 'gs_output',
                        action = 'store_true',
                        help = 'Ignored, present for compatibility only')
    parser.add_option('--question', '-q',
                        dest = 'grade_question',
                        default = None,
                        help = 'Grade only one question (e.g. `-q q1`)')
    parser.add_option('--no-graphics',
                        dest = 'no_graphics',
                        action = 'store_true',
                        help = 'Do not display graphics (visualizing your implementation is highly recommended for debugging).')
    parser.add_option('--mute',
                        dest = 'mute_output',
                        action = 'store_true',
                        help = 'Mute output from executing tests')
    parser.add_option('--check-dependencies',
                        dest = 'check_dependencies',
                        action = 'store_true',
                        help = 'check that numpy and matplotlib are installed')
    parser.add_option('--vocareumGradeFile',
                      dest='vocareumGradeFile',
                      default=None,
                      help='file of scores to display on Vocareum leaderboard')
    (options, args) = parser.parse_args(argv)
    return options

def main():
    options = parse_options(sys.argv)
    if options.check_dependencies:
        check_dependencies()
        return

    if options.no_graphics:
        disable_graphics()

    questions = set()
    maxes = {}
    for q, points, fn in TESTS:
        questions.add(q)
        maxes[q] = maxes.get(q, 0) + points
        if q not in PREREQS:
            PREREQS[q] = set()

    questions = list(sorted(questions))
    if options.grade_question:
        if options.grade_question not in questions:
            print("ERROR: question {} does not exist".format(options.grade_question))
            sys.exit(1)
        else:
            questions = [options.grade_question]
            PREREQS[options.grade_question] = set()

    tracker = Tracker(questions, maxes, PREREQS, options.mute_output, leaderboard_name="F1-score",
                      vocareum_grade_file=options.vocareumGradeFile)
    for q in questions:
        started = tracker.begin_q(q)
        if not started:
            continue

        for testq, points, fn in TESTS:
            if testq != q:
                continue
            tracker.begin_test(fn.__name__)
            try:
                fn(tracker)
            except KeyboardInterrupt:
                tracker.unmute()
                print("\n\nCaught KeyboardInterrupt: aborting autograder")
                tracker.finalize()
                print("\n[autograder was interrupted before finishing]")
                sys.exit(1)
            except:
                tracker.unmute()
                print(traceback.format_exc())
            tracker.end_test(points)
        tracker.end_q()
    tracker.finalize()

################################################################################
# Tests begin here
################################################################################

import numpy as np
import matplotlib
import contextlib

import nn
import backend

def check_dependencies():
    import matplotlib.pyplot as plt
    import time
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    line, = ax.plot([], [], color="black")
    plt.show(block=False)

    for t in range(400):
        angle = t * 0.05
        x = np.sin(angle)
        y = np.cos(angle)
        line.set_data([x,-x], [y,-y])
        fig.canvas.draw_idle()
        fig.canvas.start_event_loop(1e-3)
    import pandas as pd
    from sklearn import preprocessing


def disable_graphics():
    backend.use_graphics = False

@contextlib.contextmanager
def no_graphics():
    old_use_graphics = backend.use_graphics
    backend.use_graphics = False
    yield
    backend.use_graphics = old_use_graphics

def verify_node(node, expected_type, expected_shape, method_name):
    if expected_type == 'parameter':
        assert node is not None, (
            "{} should return an instance of nn.Parameter, not None".format(method_name))
        assert isinstance(node, nn.Parameter), (
            "{} should return an instance of nn.Parameter, instead got type {!r}".format(
            method_name, type(node).__name__))
    elif expected_type == 'loss':
        assert node is not None, (
            "{} should return an instance a loss node, not None".format(method_name))
        assert isinstance(node, (nn.SquareLoss, nn.SoftmaxLoss)), (
            "{} should return a loss node, instead got type {!r}".format(
            method_name, type(node).__name__))
    elif expected_type == 'node':
        assert node is not None, (
            "{} should return a node object, not None".format(method_name))
        assert isinstance(node, nn.Node), (
            "{} should return a node object, instead got type {!r}".format(
            method_name, type(node).__name__))
    else:
        assert False, "If you see this message, please report a bug in the autograder"

    if expected_type != 'loss':
        assert all([(expected is '?' or actual == expected) for (actual, expected) in zip(node.data.shape, expected_shape)]), (
            "{} should return an object with shape {}, got {}".format(
                method_name, nn.format_shape(expected_shape), nn.format_shape(node.data.shape)))

def trace_node(node_to_trace):
    """
    Returns a set containing the node and all ancestors in the computation graph
    """
    nodes = set()
    tape = []

    def visit(node):
        if node not in nodes:
            for parent in node.parents:
                visit(parent)
            nodes.add(node)
            tape.append(node)

    visit(node_to_trace)

    return nodes

@test('q1', points=6)
def check_perceptron(tracker):
    import models

    print("Sanity checking perceptron...")
    np_random = np.random.RandomState(0)
    # Check that the perceptron weights are initialized to a vector with `dimensions` entries.
    for dimensions in range(1, 10):
        p = models.PerceptronModel(dimensions)
        p_weights = p.get_weights()
        verify_node(p_weights, 'parameter', (1, dimensions), "PerceptronModel.get_weights()")

    # Check that run returns a node, and that the score in the node is correct
    for dimensions in range(1, 10):
        p = models.PerceptronModel(dimensions)
        p_weights = p.get_weights()
        verify_node(p_weights, 'parameter', (1, dimensions), "PerceptronModel.get_weights()")
        point = np_random.uniform(-10, 10, (1, dimensions))
        score = p.run(nn.Constant(point))
        verify_node(score, 'node', (1, 1), "PerceptronModel.run()")
        calculated_score = nn.as_scalar(score)
        expected_score = float(np.dot(point.flatten(), p_weights.data.flatten()))
        assert np.isclose(calculated_score, expected_score), (
            "The score computed by PerceptronModel.run() ({:.4f}) does not match the expected score ({:.4f})".format(
            calculated_score, expected_score))

    # Check that get_prediction returns the correct values, including the
    # case when a point lies exactly on the decision boundary
    for dimensions in range(1, 10):
        p = models.PerceptronModel(dimensions)
        random_point = np_random.uniform(-10, 10, (1, dimensions))
        for point in (random_point, np.zeros_like(random_point)):
            prediction = p.get_prediction(nn.Constant(point))
            assert prediction == 1 or prediction == -1, (
                "PerceptronModel.get_prediction() should return 1 or -1, not {}".format(
                prediction))

            expected_prediction = np.asscalar(np.where(np.dot(point, p.get_weights().data.T) >= 0, 1, -1))
            assert prediction == expected_prediction, (
                "PerceptronModel.get_prediction() returned {}; expected {}".format(
                    prediction, expected_prediction))

    tracker.add_points(2) # Partial credit for passing sanity checks

    print("Sanity checking perceptron weight updates...")

    # Test weight updates. This involves constructing a dataset that
    # requires 0 or 1 updates before convergence, and testing that weight
    # values change as expected. Note that (multiplier < -1 or multiplier > 1)
    # must be true for the testing code to be correct.
    dimensions = 2
    for multiplier in (-5, -2, 2, 5):
        p = models.PerceptronModel(dimensions)
        orig_weights = p.get_weights().data.reshape((1, dimensions)).copy()
        if np.abs(orig_weights).sum() == 0.0:
            # This autograder test doesn't work when weights are exactly zero
            continue
        point = multiplier * orig_weights
        sanity_dataset = backend.Dataset(
            x=np.tile(point, (500, 1)),
            y=np.ones((500, 1)) * -1.0
        )
        p.train(sanity_dataset)
        new_weights = p.get_weights().data.reshape((1, dimensions))

        if multiplier < 0:
            expected_weights = orig_weights
        else:
            expected_weights = orig_weights - point

        if not np.all(new_weights == expected_weights):
            print()
            print("Initial perceptron weights were: [{:.4f}, {:.4f}]".format(
                orig_weights[0,0], orig_weights[0,1]))
            print("All data points in the dataset were identical and had:")
            print("    x = [{:.4f}, {:.4f}]".format(
                point[0,0], point[0,1]))
            print("    y = -1")
            print("Your trained weights were: [{:.4f}, {:.4f}]".format(
                new_weights[0,0], new_weights[0,1]))
            print("Expected weights after training: [{:.4f}, {:.4f}]".format(
                expected_weights[0,0], expected_weights[0,1]))
            print()
            assert False, "Weight update sanity check failed"

    print("Sanity checking complete. Now training perceptron")
    model = models.PerceptronModel(3)
    dataset = backend.PerceptronDataset(model)

    model.train(dataset)
    backend.maybe_sleep_and_close(1)

    assert dataset.epoch != 0, "Perceptron code never iterated over the training data"

    accuracy = np.mean(np.where(np.dot(dataset.x, model.get_weights().data.T) >= 0.0, 1.0, -1.0) == dataset.y)
    if accuracy < 1.0:
        print("The weights learned by your perceptron correctly classified {:.2%} of training examples".format(accuracy))
        print("To receive full points for this question, your perceptron must converge to 100% accuracy")
        return

    tracker.add_points(4)

@test('q2', points=6)
def check_logistic_regression(tracker):
    import models
    model = models.LogisticRegressionModel(1)
    dataset = backend.LogisticRegressionDataset(model)

    detected_parameters = None
    for batch_size in (1, 2, 4):
        inp_x = nn.Constant(dataset.x[:batch_size])
        inp_y = nn.Constant(dataset.y[:batch_size])
        output_node = model.run(inp_x)
        verify_node(output_node, 'node', (batch_size, 1), "RegressionModel.run()")
        trace = trace_node(output_node)
        assert inp_x in trace, "Node returned from LogisticRegressionModel.run() does not depend on the provided input (x)"

        if detected_parameters is None:
            detected_parameters = [node for node in trace if isinstance(node, nn.Parameter)]

        for node in trace:
            assert not isinstance(node, nn.Parameter) or node in detected_parameters, (
                "Calling LogisticRegressionModel.run() multiple times should always re-use the same parameters, but a new nn.Parameter object was detected")

    tracker.add_points(2)  # Partial credit for passing sanity checks
    model.train(dataset, 0.01, 2000)
    backend.maybe_sleep_and_close(1)

    predicted = np.array([nn.as_scalar(model.run(nn.Constant(i.reshape(1, 1)))) for i in dataset.x])
    predictions = np.array([model.get_prediction(nn.Constant(i.reshape(1, 1))) for i in dataset.x])
    expected_predictions = predictions >= 0.5
    if (expected_predictions != predictions).any():
        print('model.get_prediction does not match model.run results')
    else:
        tracker.add_points(2)

    y = dataset.y.flatten()
    log_likelihood = np.sum(y * np.log(predicted) + (1 - y) * np.log(1 - predicted))

    log_likelihood_threshold = -60
    if log_likelihood >= log_likelihood_threshold:
        print("Your final log-likelihood is: {:f} (threshold is {:f})".format(log_likelihood, log_likelihood_threshold))
        tracker.add_points(2)
    else:
        print("Your final log-likelihood ({:f}) must be greater than {:f} to receive full points for this question"
              "".format(log_likelihood, log_likelihood_threshold))

@test('q3', points=6)
def check_digit_classification(tracker):
    import models
    from mnist_config import student_config as config
    EXPECTED_KEYS = ['input_dim', 'output_dim', 'hidden_dim', 'layers', 'epochs', 'learning_rate', 'batch_size']
    for key in EXPECTED_KEYS:
        assert key in config, f"Expected key {key} to be in the student_config."
    model = models.ClassificationModel(input_dim=config['input_dim'], output_dim=config['output_dim'],
                                       hidden_dim=config['hidden_dim'], layers=config['layers'])
    dataset = backend.DigitClassificationDataset(model)

    detected_parameters = None
    for batch_size in (1, 2, 4):
        inp_x = nn.Constant(dataset.x[:batch_size])
        inp_y = nn.Constant(dataset.y[:batch_size])
        output_node = model.run(inp_x)
        verify_node(output_node, 'node', (batch_size, 10), "DigitClassificationModel.run()")
        trace = trace_node(output_node)
        assert inp_x in trace, "Node returned from DigitClassificationModel.run() does not depend on the provided input (x)"

        if detected_parameters is None:
            detected_parameters = [node for node in trace if isinstance(node, nn.Parameter)]

        for node in trace:
            assert not isinstance(node, nn.Parameter) or node in detected_parameters, (
                "Calling DigitClassificationModel.run() multiple times should always re-use the same parameters, but a new nn.Parameter object was detected")

    for batch_size in (1, 2, 4):
        inp_x = nn.Constant(dataset.x[:batch_size])
        inp_y = nn.Constant(dataset.y[:batch_size])
        loss_node = model.get_loss(inp_x, inp_y)
        verify_node(loss_node, 'loss', None, "DigitClassificationModel.get_loss()")
        trace = trace_node(loss_node)
        assert inp_x in trace, "Node returned from DigitClassificationModel.get_loss() does not depend on the provided input (x)"
        assert inp_y in trace, "Node returned from DigitClassificationModel.get_loss() does not depend on the provided labels (y)"

        for node in trace:
            assert not isinstance(node, nn.Parameter) or node in detected_parameters, (
                "DigitClassificationModel.get_loss() should not use additional parameters not used by DigitClassificationModel.run()")

    tracker.add_points(2)  # Partial credit for passing sanity checks

    model.train(dataset, learning_rate=config['learning_rate'], batch_size=config['batch_size'],
                epochs=config['epochs'])

    test_logits = model.run(nn.Constant(dataset.test_images)).data
    test_predicted = np.argmax(test_logits, axis=1)
    test_accuracy = np.mean(test_predicted == dataset.test_labels)

    accuracy_threshold = 0.97
    if test_accuracy >= accuracy_threshold:
        print("Your final test set accuracy is: {:%}".format(test_accuracy))
        tracker.add_points(4)
    else:
        print("Your final test set accuracy ({:%}) must be at least {:.0%} to receive full points for this question".format(test_accuracy, accuracy_threshold))

    if dataset.best_model is not None:
        best_model_logits = dataset.best_model.run(nn.Constant(dataset.test_images)).data
        best_model_predicted = np.argmax(best_model_logits, axis=1)
        best_model_acc = np.mean(best_model_predicted == dataset.test_labels)
        print("With early stopping, your score would be {:%}".format(best_model_acc))
# @test('q4', points=7)
# def check_lang_id(tracker):
#     import models
#     model = models.LanguageIDModel()
#     dataset = backend.LanguageIDDataset(model)
#
#     detected_parameters = None
#     for batch_size, word_length in ((1, 1), (2, 1), (2, 6), (4, 8)):
#         start = dataset.dev_buckets[-1, 0]
#         end = start + batch_size
#         inp_xs, inp_y = dataset._encode(dataset.dev_x[start:end], dataset.dev_y[start:end])
#         inp_xs = inp_xs[:word_length]
#
#         output_node = model.run(inp_xs)
#         verify_node(output_node, 'node', (batch_size, len(dataset.language_names)), "LanguageIDModel.run()")
#         trace = trace_node(output_node)
#         for inp_x in inp_xs:
#             assert inp_x in trace, "Node returned from LanguageIDModel.run() does not depend on all of the provided inputs (xs)"
#
#         # Word length 1 does not use parameters related to transferring the
#         # hidden state across timesteps, so initial parameter detection is only
#         # run for longer words
#         if word_length > 1:
#             if detected_parameters is None:
#                 detected_parameters = [node for node in trace if isinstance(node, nn.Parameter)]
#
#             for node in trace:
#                 assert not isinstance(node, nn.Parameter) or node in detected_parameters, (
#                     "Calling LanguageIDModel.run() multiple times should always re-use the same parameters, but a new nn.Parameter object was detected")
#
#     for batch_size, word_length in ((1, 1), (2, 1), (2, 6), (4, 8)):
#         start = dataset.dev_buckets[-1, 0]
#         end = start + batch_size
#         inp_xs, inp_y = dataset._encode(dataset.dev_x[start:end], dataset.dev_y[start:end])
#         inp_xs = inp_xs[:word_length]
#         loss_node = model.get_loss(inp_xs, inp_y)
#         trace = trace_node(loss_node)
#         for inp_x in inp_xs:
#             assert inp_x in trace, "Node returned from LanguageIDModel.run() does not depend on all of the provided inputs (xs)"
#         assert inp_y in trace, "Node returned from LanguageIDModel.get_loss() does not depend on the provided labels (y)"
#
#         for node in trace:
#             assert not isinstance(node, nn.Parameter) or node in detected_parameters, (
#                 "LanguageIDModel.get_loss() should not use additional parameters not used by LanguageIDModel.run()")
#
#     tracker.add_points(2) # Partial credit for passing sanity checks
#
#     model.train(dataset)
#
#     test_predicted_probs, test_predicted, test_correct = dataset._predict('test')
#     test_accuracy = np.mean(test_predicted == test_correct)
#     accuracy_threshold = 0.81
#     if test_accuracy >= accuracy_threshold:
#         print("Your final test set accuracy is: {:%}".format(test_accuracy))
#         tracker.add_points(5)
#     else:
#         print("Your final test set accuracy ({:%}) must be at least {:.0%} to receive full points for this question".format(test_accuracy, accuracy_threshold))


@test('q4', points=4)
def check_grid_search(tracker):
    import gridSearch

    class TestGridSearch(gridSearch.GridSearch):
        def __init__(self, hyperparam_grid, best_params):
            super(TestGridSearch, self).__init__(hyperparam_grid)
            self.calls_hp = []
            self.calls_ds = []
            self.calls_eval_ds = []
            self.best_params = best_params

        def train_and_eval(self, config, train_dataset, eval_dataset):
            self.calls_hp.append(config)
            self.calls_ds.append(train_dataset)
            self.calls_eval_ds.append(eval_dataset)
            score = 1.0 if config == self.best_params else 0.0
            return config, score

    HPS = {
        'learning_rate': [0.1, 0.01],
        'batch_size': [32, 64],
        'epochs': [10,20]
    }
    BEST_HPS = {
        'learning_rate': 0.1,
        'batch_size': 64,
        'epochs': 20
    }
    DATASET = [0,1,2,3,4]
    DATASET_EVAL = [5,6]
    EXPECTED_CALLS = [{'batch_size': 32, 'epochs': 10, 'learning_rate': 0.1}, {'batch_size': 32, 'epochs': 10, 'learning_rate': 0.01}, {'batch_size': 32, 'epochs': 20, 'learning_rate': 0.1}, {'batch_size': 32, 'epochs': 20, 'learning_rate': 0.01}, {'batch_size': 64, 'epochs': 10, 'learning_rate': 0.1}, {'batch_size': 64, 'epochs': 10, 'learning_rate': 0.01}, {'batch_size': 64, 'epochs': 20, 'learning_rate': 0.1}, {'batch_size': 64, 'epochs': 20, 'learning_rate': 0.01}]
    test_grid = TestGridSearch(HPS, BEST_HPS)
    best_model, best_score, best_config = test_grid.perform_grid_search(DATASET, DATASET_EVAL)
    for config in EXPECTED_CALLS:
        assert config in test_grid.calls_hp, f'Call with hyperparameters {config} not made'
    assert len(test_grid.calls_hp) == len(EXPECTED_CALLS), \
        f'Number of training calls {len(test_grid.calls_hp)}, {len(EXPECTED_CALLS)} expected'
    for ds in test_grid.calls_ds:
        assert ds == DATASET, 'Called train_and_eval with incorrect training dataset'
    for ds in test_grid.calls_eval_ds:
        assert ds == DATASET_EVAL, 'Called train_and_eval with incorrect eval dataset'

    assert best_model == BEST_HPS, "Did not return best model"
    assert best_score == 1, "Did not return best score"
    assert best_config == BEST_HPS, "Did not return the best config"
    tracker.add_points(2)

    HP_CONFIG = {
        'input_dim': 5,
        'output_dim': 5,
        'hidden_dim': 16,
        'layers': 3,
        'learning_rate': 0.01,
        'epochs': 10,
        'batch_size': 10
    }

    class ModelRecorder(object):
        def __init__(self, input_dim=784, hidden_dim=4, layers=2, output_dim=10):
            """Initialize your model parameters here"""
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.layers = layers
            self.output_dim = output_dim
            self.batch_size = None
            self.epochs = None
            self.learning_rate = None
            self.train_ds = None
            self.eval_ds = None

        def train(self, dataset, learning_rate, epochs, batch_size):
            self.train_ds = dataset
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size

        def eval(self, eval_dataset):
            self.eval_ds = eval_dataset
            return 2.0

    gridSearch.ClassificationModel = ModelRecorder  # inject ModelRecorder into gridSearch module
    grid_searcher = gridSearch.GridSearch({})
    model, score = grid_searcher.train_and_eval(HP_CONFIG, DATASET, DATASET_EVAL)

    for key in ['input_dim', 'output_dim', 'hidden_dim', 'layers']:
        assert HP_CONFIG[key] == model.__dict__[key], f'Expected train_and_eval to instantiate a ClassificationModel ' \
                                                      f'with the {key} provided in config'

    for key in ['epochs', 'batch_size', 'learning_rate']:
        assert HP_CONFIG[key] == model.__dict__[key], f'Expected train_and_eval to train a ClassificationModel ' \
                                                      f'with the {key} provided in config'

    assert model.train_ds == DATASET, "Expected train_and_eval to use the passed in train_dataset for training"

    assert model.eval_ds == DATASET_EVAL, "Expected train_and_eval to use the passed in eval_dataset for evaluation"

    assert score == 2.0, "Expected train_and_eval to return the score from model.eval"
    assert isinstance(model, ModelRecorder), "Expected train_and_eval to return a model of type ClassificationModel"

    tracker.add_points(2)

    from importlib import reload
    reload(gridSearch)  # Undo the import injection into gridSearch.ClassificationModel


@test('q5', points=2)
def evaluation_metrics(tracker):
    from evaluation import get_eval_metrics

    acc, prec, recall = get_eval_metrics([1,1], [1,0])
    assert acc == 0.5 and prec == 0.5 and recall == 1.0, 'failing for pred = [1,1] and labels = [1,0]'

    acc, prec, recall = get_eval_metrics([0, 1], [1, 0])
    assert acc == 0.0 and prec == 0.0 and recall == 0.0, 'failing for pred = [0,1] and labels = [1,0]'

    acc, prec, recall = get_eval_metrics([1, 0], [1, 0])
    assert acc == 1.0 and prec == 1.0 and recall == 1.0, 'failing for pred = [1,0] and labels = [1,0]'

    acc, prec, recall = get_eval_metrics([1, 0], [1, 1])
    assert acc == 0.5 and prec == 1.0 and recall == 0.5, 'failing for pred = [1,0] and labels = [1,1]'

    tracker.add_points(2)


@test('q6', points=2)
def mental_health_survey(tracker):
    from gridSearch import GridSearch
    from survey_hyperparameters import survey_hyperparameters
    from backend import MentalHealthTreatmentDataset
    from evaluation import get_eval_metrics

    TRAIN_DATASET = MentalHealthTreatmentDataset()
    EVAL_DATASET = TRAIN_DATASET.get_validation_dataset()
    TEST_DATASET = TRAIN_DATASET.get_test_dataset()

    grid_search = GridSearch(survey_hyperparameters)

    assert len(grid_search.grid_search_configurations()) > 1, "You should use survey_hyperparameters to indicate a " \
                                                              "grid of configurations to search over " \
                                                              "(must be more than 1)"

    best_model, best_eval, best_config = grid_search.perform_grid_search(TRAIN_DATASET, EVAL_DATASET)

    test_preds = np.argmax(best_model.run(nn.Constant(TEST_DATASET.x)).data, axis=1).tolist()
    test_labels = np.argmax(TEST_DATASET.y, axis=1).tolist()

    test_acc, test_prec, test_recall = get_eval_metrics(test_preds, test_labels)

    assert test_acc > 0.78, f"Test accuracy of {test_acc} does not meet 78% threshold"

    print(f"Your model's results are\n"
          f"Accuracy: {test_acc}\n"
          f"Precision: {test_prec}\n"
          f"Recall: {test_recall}\n")

    tracker.add_points(2)


@test('bonus1', points=1)
def bonus_question_analysis(tracker: Tracker):
    import bonusQuestionAnalysis as analysis

    assert analysis.student_answer_1 in analysis.legal_answers_1, \
        "Your answer must be one of 'accuracy', 'precision', 'recall'"

    assert analysis.student_answer_2 in analysis.legal_answers_2, \
        "Your answer must be one of 'knn', 'logistic regression', 'decision tree', or 'neural network'"

    # Only grade if submitting to vocareum
    if tracker.vocareum_grade_file is not None:
        CORRECT_ANSWER_1 = np.load('bonus_question_solution_1.npy').tolist()
        if analysis.student_answer_1 == CORRECT_ANSWER_1:
            tracker.add_points(0.5)
        CORRECT_ANSWER_2 = np.load('bonus_question_solution_2.npy').tolist()
        if analysis.student_answer_2 == CORRECT_ANSWER_2:
            tracker.add_points(0.5)
    else:
        print("This question will only be graded on vocareum.")


@test('bonus2', points=3)
def bonus_question_open_ended(tracker: Tracker):
    import heart_disease as hd
    train_data = hd.read_data(path='data/heart_2020_cleaned.csv')
    test_data = hd.read_data(path='data/heart_2020_cleaned.csv', split=slice(50000,60000))

    EXPECTED_LABELS = train_data['HeartDisease']
    train_data = hd.process_data(train_data)
    assert EXPECTED_LABELS.tolist() == train_data['HeartDisease'].tolist(), 'process_data should not change the labels'
    X_train, X_val, y_train, y_val = hd.split_data(train_data)

    test_data = hd.process_data(test_data)
    X_test = test_data.drop(columns=['HeartDisease'], axis=1)
    y_test = test_data['HeartDisease']

    y_pred = hd.run_student_model(X_train, X_val, y_train, y_val, X_test=X_test)

    from sklearn import metrics
    score = metrics.f1_score(y_test, y_pred)
    LOW_THRESHOLD = 0.2
    HIGH_THRESHOLD = 0.4
    rescaled = (score - LOW_THRESHOLD)/(HIGH_THRESHOLD - LOW_THRESHOLD)
    bounded = min(1, max(0, rescaled))
    POINTS_AVAILABLE = 3
    grade = POINTS_AVAILABLE * bounded
    print(f"Your f1 score is {score}")
    tracker.add_points(grade)
    tracker.set_leaderboard(score)








if __name__ == '__main__':
    main()
