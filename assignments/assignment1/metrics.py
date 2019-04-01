def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    cnt = prediction.shape[0]
    #print(prediction.shape)
    true_positive = 0
    true_total = 0
    positive_total = 0
    for i in range(cnt):
        if ground_truth[i]: true_total += 1
        if prediction[i]: positive_total += 1
        if prediction[i] == ground_truth[i] and prediction[i]: true_positive += 1
        if prediction[i] == ground_truth[i]: accuracy += 1

    recall = true_positive / true_total
    precision = true_positive / positive_total
    accuracy = accuracy / cnt
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return 0
