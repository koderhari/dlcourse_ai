import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    predictions_copy = predictions.copy()
    predictions_copy -= np.max(predictions_copy)
    e = np.exp(predictions_copy)
    if len(predictions.shape) == 2:
        denom = np.sum(e, axis=1).shape(predictions.shape[0],1)  #
    else:
        denom = np.sum(e)

    return e / denom


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    if type(target_index) is int:
        return -np.log(probs[target_index])
    else:
        return -np.sum(np.log(probs[target_index])) / target_index.shape(0)     # среднее значение

    #raise Exception("Not implemented!")


def softmax_with_cross_entropy(predictions, target_index):
    '''
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    '''
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    #raise Exception("Not implemented!")
    print(predictions)
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    print(loss)
    d_soft = -1 / probs[target_index]
    print(d_soft)
    predictions_copy = predictions.copy()
    predictions_copy -= np.max(predictions_copy)
    denom = np.sum(np.exp(predictions_copy)) #axis=1

    #print(denom.shape) == target_index.shape
    #print(probs)
#ты кусок гавна зачем probs возводишь в exp это уже наши предикшины возведенные в экспоненту это уже softmax!!!
    #надо с предикшинами работать
    exp = np.exp(predictions_copy)
    print(predictions_copy)
    print(exp)
    print(denom)
    #print(denom)
    #в зависимости от таргета индекса производная будет разная
    #print(exp)
    #d_z = (exp * denom - exp * exp) / (denom * denom)
    S = exp / denom
    print(S)
    #print(S)
    #print(predictions.shape)
    #print(S.shape) # must be as predictions or probs shape
    print('---')
    #d_z = S * (1 - S) # для таргет индексов
    d_z = - S[target_index] * S
    d_z[target_index] += S[target_index] # S_i^2 - S_i*S_j
    #d_z = S
    # -S_i*S_j для не таргет
    #print(d_soft * S * (1 - S))
    #print(d_z.shape)
    dprediction = d_soft * d_z
    #print(dprediction) must have shape as predictions
    #print(dprediction)
    return loss, dprediction


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")

    return loss, grad
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W)
    # здесь просто вызвать софтмак виз кросс энтропи
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loops
    raise Exception("Not implemented!")
    
    return loss, dW


class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)

        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            #возможно здесь будет 0 вместо первого параметра batch_size?
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            #здесь нюанс кэ и рег разбиты на два шага, по факту это так и есть
            #даже градиент это получается сумма производных двух функции
            #то есть вычитать из W тоже можно по очереди
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!
            raise Exception("Not implemented!")

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        # здесь я просто считаю софтмакс беру максимальный скоре и это мой предикт
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops
        raise Exception("Not implemented!")

        return y_pred



                
                                                          

            

                
