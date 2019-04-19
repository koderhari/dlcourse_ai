import numpy as np


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()
    #from_x = x.copy()
    #to_x = x.copy()

    #from_x -= tol
    #to_x += tol
    #f_x_from, foo = f(from_x)
    #f_x_to, foo = f(to_x)

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]
        from_x = orig_x.copy()
        to_x = orig_x.copy()
        #print(ix)
        #print(orig_x)
        #print(from_x)
        from_x[ix] = orig_x[ix] - tol
        to_x[ix] = orig_x[ix] + tol
        f_x_from, foo = f(from_x)
        f_x_to, foo = f(to_x)
        #print(f_x_to)
        numeric_grad_at_ix = (f_x_to - f_x_from) / (2 * tol)
        #нужно взять х прибавить-отнять эпсилон скормить функции и вычислить градиент
        # TODO compute value of numeric gradient of f to idx
        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            return False

        it.iternext()

    print("Gradient check passed!")
    return True

        

        
