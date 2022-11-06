import collections
import os
import time
import os
import copy

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import nn

from process_survey import run_data_processing

use_graphics = True

def maybe_sleep_and_close(seconds):
    if use_graphics and plt.get_fignums():
        time.sleep(seconds)
        for fignum in plt.get_fignums():
            fig = plt.figure(fignum)
            plt.close(fig)
            try:
                # This raises a TclError on some Windows machines
                fig.canvas.start_event_loop(1e-3)
            except:
                pass

def get_data_path(filename):
    path = os.path.join(
        os.path.dirname(__file__), os.pardir, "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), "data", filename)
    if not os.path.exists(path):
        path = os.path.join(
            os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        raise Exception("Could not find data file: {}".format(filename))
    return path

class Dataset(object):
    def __init__(self, x, y):
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert np.issubdtype(x.dtype, np.floating)
        assert np.issubdtype(y.dtype, np.floating)
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def pick_random(self, batch_size):
        n = self.x.shape[0]
        idxs = np.random.choice(n, size=batch_size)
        return nn.Constant(self.x[idxs]), nn.Constant(self.y[idxs])

    def iterate_once(self, batch_size):
        assert isinstance(batch_size, int) and batch_size > 0, (
            "Batch size should be a positive integer, got {!r}".format(
                batch_size))
        # TODO: Check if we need dataset divisible by batch size
        # assert self.x.shape[0] % batch_size == 0, (
        #     "Dataset size {:d} is not divisible by batch size {:d}".format(
        #         self.x.shape[0], batch_size))
        index = 0
        while index < self.x.shape[0]:
            x = self.x[index:index + batch_size]
            y = self.y[index:index + batch_size]
            yield nn.Constant(x), nn.Constant(y)
            index += batch_size

    def iterate_forever(self, batch_size):
        while True:
            yield from self.iterate_once(batch_size)

    def get_validation_accuracy(self):
        raise NotImplementedError(
            "No validation data is available for this dataset. "
            "In this assignment, only the Digit Classification and Language "
            "Identification datasets have validation data.")

    def __len__(self):
        return len(self.x)

class PerceptronDataset(Dataset):
    def __init__(self, model):
        points = 500
        x = np.hstack([np.random.randn(points, 2), np.ones((points, 1))])
        y = np.where(x[:, 0] + 2 * x[:, 1] - 1 >= 0, 1.0, -1.0)
        super().__init__(x, np.expand_dims(y, axis=1))

        self.model = model
        self.epoch = 0

        if use_graphics:
            fig, ax = plt.subplots(1, 1)
            limits = np.array([-3.0, 3.0])
            ax.set_xlim(limits)
            ax.set_ylim(limits)
            positive = ax.scatter(*x[y == 1, :-1].T, color="red", marker="+")
            negative = ax.scatter(*x[y == -1, :-1].T, color="blue", marker="_")
            line, = ax.plot([], [], color="black")
            text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")
            ax.legend([positive, negative], [1, -1])
            plt.show(block=False)

            self.fig = fig
            self.limits = limits
            self.line = line
            self.text = text
            self.last_update = time.time()

    def iterate_once(self, batch_size):
        self.epoch += 1

        for i, (x, y) in enumerate(super().iterate_once(batch_size)):
            yield x, y

            if use_graphics and time.time() - self.last_update > 0.01:
                w = self.model.get_weights().data.flatten()
                limits = self.limits
                if w[1] != 0:
                    self.line.set_data(limits, (-w[0] * limits - w[2]) / w[1])
                elif w[0] != 0:
                    self.line.set_data(np.full(2, -w[2] / w[0]), limits)
                else:
                    self.line.set_data([], [])
                self.text.set_text(
                    "epoch: {:,}\npoint: {:,}/{:,}\nweights: {}".format(
                        self.epoch, i * batch_size + 1, len(self.x), w))
                self.fig.canvas.draw_idle()
                self.fig.canvas.start_event_loop(1e-3)
                self.last_update = time.time()

class LogisticRegressionDataset(Dataset):
    def __init__(self, model):
        x = np.expand_dims(np.linspace(-2 * np.pi, 2 * np.pi, num=200), axis=1)
        np.random.RandomState(0).shuffle(x)
        self.argsort_x = np.argsort(x.flatten())
        y = []
        for x_val in x:
            prob = 1 / (1 + np.exp(x_val))  # backwards logistic sampling
            y.append(np.random.binomial(1, prob, size=1))
        y = np.array(y).astype('float')
        super().__init__(x, y)

        self.model = model
        self.processed = 0

        if use_graphics:
            fig, ax = plt.subplots(1, 1)
            ax.set_xlim(-2 * np.pi, 2 * np.pi)
            ax.set_ylim(-1.4, 1.8)
            real, = ax.plot(x[self.argsort_x], y[self.argsort_x], 'o', color="blue")
            learned, = ax.plot([], [], color="red")
            text = ax.text(0.03, 0.97, "", transform=ax.transAxes, va="top")
            ax.legend([real, learned], ["real", "learned"])
            plt.show(block=False)

            self.fig = fig
            self.learned = learned
            self.text = text
            self.last_update = time.time()

    def pick_random(self, batch_size):

        if use_graphics and self.processed % 10 == 0:
            predicted = np.array([nn.as_scalar(self.model.run(nn.Constant(i.reshape(1,1)))) for i in self.x])
            preds = predicted >= 0.5
            y = self.y.flatten()
            acc = (preds.flatten() == y).mean()
            likelihood = np.prod(y * predicted + (1 - y) * (1 - predicted))
            log_likelihood = np.sum(y * np.log(predicted) + (1 - y) * np.log(1 - predicted))
            self.learned.set_data(self.x[self.argsort_x], predicted[self.argsort_x])
            self.text.set_text("processed: {:,}\nLikelihood: {:.6f}\nLog-likelihood: {:.6f}\nAcc: {:.6f}".format(
               self.processed, likelihood, log_likelihood, acc))
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(1e-3)
            self.last_update = time.time()
        self.processed += batch_size
        if self.processed <= 100:
            time.sleep(0.01)  # slow down so students can see initial learning
        return super().pick_random(batch_size)

    def iterate_once(self, batch_size):
        for x, y in super().iterate_once(batch_size):
            yield x, y
            self.processed += batch_size

            if use_graphics and time.time() - self.last_update > 0.1:
                predicted = self.model.run(nn.Constant(self.x)).data
                loss = self.model.get_loss(
                    nn.Constant(self.x), nn.Constant(self.y)).data
                self.learned.set_data(self.x[self.argsort_x], predicted[self.argsort_x])
                self.text.set_text("processed: {:,}\nloss: {:.6f}".format(
                   self.processed, loss))
                self.fig.canvas.draw_idle()
                self.fig.canvas.start_event_loop(1e-3)
                self.last_update = time.time()

class DigitClassificationDataset(Dataset):
    def __init__(self, model):
        mnist_path = get_data_path("mnist.npz")

        with np.load(mnist_path) as data:
            self.train_images = train_images = data["train_images"]
            self.train_labels = train_labels = data["train_labels"]
            test_images = data["test_images"]
            test_labels = data["test_labels"]
            assert len(train_images) == len(train_labels) == 60000
            assert len(test_images) == len(test_labels) == 10000
            self.dev_images = test_images[0::2]
            self.dev_labels = test_labels[0::2]
            self.test_images = test_images[1::2]
            self.test_labels = test_labels[1::2]

        train_labels_one_hot = np.zeros((len(train_images), 10))
        train_labels_one_hot[range(len(train_images)), train_labels] = 1

        super().__init__(train_images, train_labels_one_hot)

        self.model = model
        self.epoch = 0

        if use_graphics:
            width = 20  # Width of each row expressed as a multiple of image width
            samples = 100  # Number of images to display per label
            fig = plt.figure()
            ax = {}
            images = collections.defaultdict(list)
            texts = collections.defaultdict(list)
            for i in reversed(range(10)):
                ax[i] = plt.subplot2grid((30, 1), (3 * i, 0), 2, 1,
                                         sharex=ax.get(9))
                plt.setp(ax[i].get_xticklabels(), visible=i == 9)
                ax[i].set_yticks([])
                ax[i].text(-0.03, 0.5, i, transform=ax[i].transAxes,
                           va="center")
                ax[i].set_xlim(0, 28 * width)
                ax[i].set_ylim(0, 28)
                for j in range(samples):
                    images[i].append(ax[i].imshow(
                        np.zeros((28, 28)), vmin=0, vmax=1, cmap="Greens",
                        alpha=0.3))
                    texts[i].append(ax[i].text(
                        0, 0, "", ha="center", va="top", fontsize="smaller"))
            ax[9].set_xticks(np.linspace(0, 28 * width, 11))
            ax[9].set_xticklabels(
                ["{:.1f}".format(num) for num in np.linspace(0, 1, 11)])
            ax[9].tick_params(axis="x", pad=16)
            ax[9].set_xlabel("Probability of Correct Label")
            status = ax[0].text(
                0.5, 1.5, "", transform=ax[0].transAxes, ha="center",
                va="bottom")
            plt.show(block=False)

            self.width = width
            self.samples = samples
            self.fig = fig
            self.images = images
            self.texts = texts
            self.status = status
            self.last_update = time.time()
        self.best_model = None

    def iterate_once(self, batch_size):
        self.epoch += 1

        for i, (x, y) in enumerate(super().iterate_once(batch_size)):
            yield x, y

            best_dev_accuracy = 0

            if use_graphics and time.time() - self.last_update > 2:
                sample_train = np.random.choice(self.x.shape[0], size=5000)
                sample_train_images = self.x[sample_train]
                sample_train_labels = self.train_labels[sample_train]
                sample_train_logits = self.model.run(nn.Constant(sample_train_images)).data
                sample_train_predicted = np.argmax(sample_train_logits, axis=1)
                sample_train_accuracy = np.mean(sample_train_predicted == sample_train_labels)
                dev_logits = self.model.run(nn.Constant(self.dev_images)).data
                dev_predicted = np.argmax(dev_logits, axis=1)
                dev_probs = np.exp(nn.SoftmaxLoss.log_softmax(dev_logits))
                dev_accuracy = np.mean(dev_predicted == self.dev_labels)

                if dev_accuracy > best_dev_accuracy:
                    best_dev_accuracy = dev_accuracy
                    self.best_model = copy.deepcopy(self.model)

                self.status.set_text(
                    "epoch: {:d}, batch: {:d}/{:d}, validation accuracy: "
                    "{:.2%}, train accuracy: {:.2%}".format(
                        self.epoch, i, len(self.x) // batch_size, dev_accuracy, sample_train_accuracy))
                for i in range(10):
                    predicted = dev_predicted[self.dev_labels == i]
                    probs = dev_probs[self.dev_labels == i][:, i]
                    linspace = np.linspace(
                        0, len(probs) - 1, self.samples).astype(int)
                    indices = probs.argsort()[linspace]
                    for j, (prob, image) in enumerate(zip(
                            probs[indices],
                            self.dev_images[self.dev_labels == i][indices])):
                        self.images[i][j].set_data(image.reshape((28, 28)))
                        left = prob * (self.width - 1) * 28
                        if predicted[indices[j]] == i:
                            self.images[i][j].set_cmap("Greens")
                            self.texts[i][j].set_text("")
                        else:
                            self.images[i][j].set_cmap("Reds")
                            self.texts[i][j].set_text(predicted[indices[j]])
                            self.texts[i][j].set_x(left + 14)
                        self.images[i][j].set_extent([left, left + 28, 0, 28])
                self.fig.canvas.draw_idle()
                self.fig.canvas.start_event_loop(1e-3)
                self.last_update = time.time()

    def get_validation_accuracy(self):
        dev_logits = self.model.run(nn.Constant(self.dev_images)).data
        dev_predicted = np.argmax(dev_logits, axis=1)
        dev_accuracy = np.mean(dev_predicted == self.dev_labels)
        return dev_accuracy


class MentalHealthTreatmentDataset(Dataset):
    def __init__(self, copy=None):
        # run_data_processing(get_data_path("survey.csv"), 'data/survey.npz')
        # run_data_processing(get_data_path('survey_test.csv'), 'data/survey_test.npz')
        if copy is None:
            data_path = get_data_path("survey.npz")

            with np.load(data_path) as data:
                x = data["x"]
                y = data["y"]
                assert len(x) == len(y)
                y = np.concatenate([np.logical_not(y).astype('float'), y], axis=1)

                data_train, data_test, labels_train, labels_test = train_test_split(x, y, test_size=0.20, random_state=42)
                data_eval, data_test, labels_eval, labels_test = train_test_split(data_test, labels_test, test_size=0.5, random_state=42)

                self.train_x = self.x = data_train
                self.train_y = self.y = labels_train
                self.eval_x = data_eval
                self.eval_y = labels_eval
                self.test_x = data_test
                self.test_y = labels_test

        else:
            self.x = copy.x.copy()
            self.y = copy.y.copy()
            self.train_x = copy.train_x.copy()
            self.train_y = copy.train_y.copy()
            self.eval_x = copy.eval_x.copy()
            self.eval_y = copy.eval_y.copy()
            self.test_x = copy.test_x.copy()
            self.test_y = copy.test_y.copy()

        super().__init__(self.x, self.y)
        self.epoch = 0

    def iterate_once(self, batch_size):
        self.epoch += 1

        for i, (x, y) in enumerate(super().iterate_once(batch_size)):
            yield x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            ds = MentalHealthTreatmentDataset(copy=self)
            ds.x = ds.x[key]
            ds.y = ds.y[key]
            return ds
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index ({key}) is out of range.")
            return self.x[key], self.y[key]  # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type.")

    def __add__(self, other):
        ds = MentalHealthTreatmentDataset(copy=self)
        ds.x = np.concatenate([ds.x, other.x], axis=0)
        ds.y = np.concatenate([ds.y, other.y], axis=0)
        return ds

    def get_test_dataset(self):
        ds = MentalHealthTreatmentDataset(copy=self)
        ds.x = ds.test_x
        ds.y = ds.test_y
        return ds

    def get_validation_dataset(self):
        ds = MentalHealthTreatmentDataset(copy=self)
        ds.x = ds.eval_x
        ds.y = ds.eval_y
        return ds


def main():
    import models
    # model = models.PerceptronModel(3)
    # dataset = PerceptronDataset(model)
    # model.train(dataset)

    # model = models.RegressionModel()
    # dataset = RegressionDataset(model)
    # model.train(dataset)

    # model = models.DigitClassificationModel()
    # dataset = DigitClassificationDataset(model)
    # model.train(dataset)

    model = models.MentalHealthTreatmentModel({})
    dataset = MentalHealthTreatmentDataset()
    breakpoint()


if __name__ == "__main__":
    main()
