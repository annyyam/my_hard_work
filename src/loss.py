import numpy as np

def cross_entropy_loss(predictions, targets):
    n_samples = predictions.shape[0]
    log_p = -np.log(predictions[range(n_samples), targets])
    loss = np.sum(log_p) / n_samples
    return loss

def cross_entropy_grad(predictions, targets):
    n_samples = predictions.shape[0]
    grad = predictions.copy()
    grad[range(n_samples), targets] -= 1
    grad = grad / n_samples
    return grad