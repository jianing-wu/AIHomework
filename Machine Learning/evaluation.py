
def get_eval_metrics(predicted_classes, true_class_labels):
    """
    Returns evaluation metrics for binary classification problem.
    (1 is the positive class, 0 is the negative class)

    :predicted_classes: a list of binary class predictions (1 or 0)
    :true_class_labels: a list of true classes (1 or 0)

    :returns: accuracy, precision, recall
    """
    correct = 0
    total = len(true_class_labels)
    truePos = 0
    falsePos = 0
    falseNeg = 0
    for i in range(total):
        correct += (predicted_classes[i] == true_class_labels[i])
        truePos += (predicted_classes[i] and true_class_labels[i])
        falsePos += (predicted_classes[i] and not true_class_labels[i])
        falseNeg += (not predicted_classes[i] and true_class_labels[i])

    acc = 1.0 * correct / total
    precision = 1.0 * truePos / (truePos + falsePos)
    recall = 1.0 * truePos / (truePos + falseNeg)

    return acc, precision, recall
