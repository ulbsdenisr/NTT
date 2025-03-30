from __future__ import print_function
from __future__ import division

import torch
import coutils
import random
import math
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# for plotting
plt.ion()
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_toy_data(num_inputs=5, input_size=4, hidden_size=10, num_classes=3,
                 dtype=torch.float32):
  N = num_inputs
  D = input_size
  H = hidden_size
  C = num_classes
 
  # We set the random seed for repeatable experiments.
  coutils.utils.fix_random_seed()
  
  # Generate some random parameters, storing them in a dict
  params = {}
  params['W1'] = 1e-4 * torch.randn(D, H, device='cuda').to(dtype)
  params['b1'] = torch.zeros(H, device='cuda').to(dtype)
  params['W2'] = 1e-4 * torch.randn(H, C, device='cuda').to(dtype)
  params['b2'] = torch.zeros(C, device='cuda').to(dtype)

  # Generate some random inputs and labels
  toy_X = 10.0 * torch.randn(N, D, device='cuda').to(dtype)
  toy_y = torch.tensor([0, 1, 2, 2, 1], dtype=torch.int64, device='cuda')
  
  return toy_X, toy_y, params

def nn_loss_part1(params, X, y=None, reg=0.0):
    """
    The first stage of our neural network implementation: Run the forward pass
    of the network to compute the hidden layer features and classification
    scores. The network architecture should be:
    
    FC layer -> ReLU (hidden) -> FC layer (scores)

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns a tuple of:
    - scores: Tensor of shape (N, C) giving the classification scores for X
    - hidden: Tensor of shape (N, H) giving the hidden layer representation
      for each input value (after the ReLU).
    """
    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    # Compute the forward pass
    hidden = None
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an tensor of     #
    # shape (N, C).                                                             #
    #############################################################################
    # Replace "pass" statement with your code
    hidden = torch.relu(X.mm(W1) + b1)
    scores = hidden.mm(W2) + b2

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    return scores, hidden

toy_X, toy_y, params = get_toy_data()

scores, _ = nn_loss_part1(params, toy_X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = torch.tensor([
        [-3.8160e-07,  1.9975e-07,  1.0911e-07],
        [-5.0228e-08,  1.2784e-07, -5.2746e-08],
        [-5.9560e-07,  9.1178e-07,  1.1879e-06],
        [-3.2737e-08,  1.8820e-07, -2.8079e-07],
        [-1.9523e-07,  2.0502e-07, -6.0692e-08]], dtype=torch.float32, device=scores.device)
print(correct_scores)
print()

# The difference should be very small. We get < 1e-10
scores_diff = (scores - correct_scores).abs().sum().item()
print('Difference between your scores and correct scores: %.2e' % scores_diff)

def nn_loss_part2(params, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs: Same as nn_loss_part1
  
    Returns:
    If y is None, return a tensor scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    N, D = X.shape

    scores, h1 = nn_loss_part1(params, X, y, reg)
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. When you implment the regularization over W, please DO   #
    # NOT multiply the regularization term by 1/2 (no coefficient). If you are  #
    # not careful here, it is easy to run into numeric instability (Check       #
    # Numeric Stability in http://cs231n.github.io/linear-classify/).           #
    #############################################################################
    # Replace "pass" statement with your code
    # Compute softmax probabilities
    scores -= scores.max(dim=1, keepdim=True)[0]  # Numeric stability
    exp_scores = torch.exp(scores)
    softmax_probs = exp_scores / exp_scores.sum(dim=1, keepdim=True)

# Compute the loss
    loss = -torch.log(softmax_probs[torch.arange(N), y]).sum() / N
    loss += reg * (torch.sum(W1 * W1) + torch.sum(W2 * W2))

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a tensor of same size #
    #############################################################################
    # Replace "pass" statement with your code
    # Backward pass
    dscores = softmax_probs
    dscores[torch.arange(N), y] -= 1
    dscores /= N

# Compute gradients
    grads = {}
    grads['W2'] = h1.t().mm(dscores) + 2 * reg * W2
    grads['b2'] = dscores.sum(dim=0)

    dh1 = dscores.mm(W2.t())
    dh1[h1 <= 0] = 0  # ReLU backpropagation

    grads['W1'] = X.t().mm(dh1) + 2 * reg * W1
    grads['b1'] = dh1.sum(dim=0)

    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

toy_X, toy_y, params = get_toy_data()

loss, _ = nn_loss_part2(params, toy_X, toy_y, reg=0.05)
print('Your loss: ', loss.item())
correct_loss = 1.0986
print('Correct loss: ', correct_loss)
diff = (correct_loss - loss).item()

# should be very small, we get < 1e-4
print('Difference: %.4e' % diff)

def compute_numeric_gradient(f, x, h=1e-7):
  """ 
  Compute the numeric gradient of f at x using a finite differences
  approximation. We use the centered difference:
  
  df/dx ~= (f(x + h) - f(x - h)) / (2 * h)
  
  Inputs:
  - f: A function that inputs a torch tensor and returns a torch scalar
  - x: A torch tensor giving the point at which to compute the gradient

  Returns:
  - grad: A tensor of the same shape as x giving the gradient of f at x
  """ 
  fx = f(x) # evaluate function value at original point
  flat_x = x.contiguous().view(-1)
  grad = torch.zeros_like(x)
  flat_grad = grad.view(-1)
  # iterate over all indexes in x
  for i in range(flat_x.shape[0]):
    oldval = flat_x[i].item() # Store the original value
    flat_x[i] = oldval + h    # Increment by h
    fxph = f(x).item()        # Evaluate f(x + h)
    flat_x[i] = oldval - h    # Decrement by h
    fxmh = f(x).item()        # Evaluate f(x - h)
    flat_x[i] = oldval        # Restore original value

    # compute the partial derivative with centered formula
    flat_grad[i] = (fxph - fxmh) / (2 * h)

  return grad


def rel_error(x, y, eps=1e-10):
  """ returns relative error between x and y """
  top = (x - y).abs().max().item()
  bot = (x.abs() + y.abs()).clamp(min=eps).max().item()
  return top / bot

reg = 0.05
toy_X, toy_y, params = get_toy_data(dtype=torch.float64)
loss, grads = nn_loss_part2(params, toy_X, toy_y, reg=reg)

for param_name, grad in grads.items():
    param = params[param_name]
    f = lambda w: nn_loss_part2(params, toy_X, toy_y, reg=reg)[0]
    grad_numeric = compute_numeric_gradient(f, param)
    error = rel_error(grad, grad_numeric)
    print('%s max relative error: %e' % (param_name, error))


def nn_train(params, loss_func, pred_func, X, y, X_val, y_val,
             learning_rate=1e-3, learning_rate_decay=0.95,
             reg=5e-6, num_iters=100,
             batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients.
      It takes as input:
      - params: Same as input to nn_train
      - X_batch: A minibatch of inputs of shape (B, D)
      - y_batch: Ground-truth labels for X_batch
      - reg: Same as input to nn_train
      And it returns a tuple of:
        - loss: Scalar giving the loss on the minibatch
        - grads: Dictionary mapping parameter names to gradients of the loss with
          respect to the corresponding parameter.
    - pred_func: prediction function that im
    - X: A PyTorch tensor of shape (N, D) giving training data.
    - y: A PyTorch tensor f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A PyTorch tensor of shape (N_val, D) giving validation data.
    - y_val: A PyTorch tensor of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.

    Returns: A dictionary giving statistics about the training process
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in range(num_iters):
        X_batch = None
        y_batch = None

        #########################################################################
        # TODO: Create a random minibatch of training data and labels, storing  #
        # them in X_batch and y_batch respectively.                             #
        # hint: torch.randint                                                   #
        #########################################################################
        # Replace "pass" statement with your code
        indices = torch.randint(0, num_train, (batch_size,))
        X_batch = X[indices]
        y_batch = y[indices]
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        # Compute loss and gradients using the current minibatch
        loss, grads = loss_func(params, X_batch, y=y_batch, reg=reg)
        loss_history.append(loss.item())

        #########################################################################
        # TODO: Use the gradients in the grads dictionary to update the         #
        # parameters of the network (stored in the dictionary self.params)      #
        # using stochastic gradient descent. You'll need to use the gradients   #
        # stored in the grads dictionary defined above.                         #
        #########################################################################
        # Replace "pass" statement with your code
        for param_name in params:
            params[param_name] -= learning_rate * grads[param_name]
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss.item()))

        # Every epoch, check train and val accuracy and decay learning rate.
        if it % iterations_per_epoch == 0:
            # Check accuracy
            y_train_pred = pred_func(params, loss_func, X_batch)
            train_acc = (y_train_pred == y_batch).float().mean().item()
            y_val_pred = pred_func(params, loss_func, X_val)
            val_acc = (y_val_pred == y_val).float().mean().item()
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            # Decay learning rate
            learning_rate *= learning_rate_decay

    return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        'val_acc_history': val_acc_history,
    }


def nn_predict(params, loss_func, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - params: a dictionary of PyTorch Tensor that store the weights of a model.
      It should have following keys with shape
          W1: First layer weights; has shape (D, H)
          b1: First layer biases; has shape (H,)
          W2: Second layer weights; has shape (H, C)
          b2: Second layer biases; has shape (C,)
    - loss_func: a loss function that computes the loss and the gradients
    - X: A PyTorch tensor of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A PyTorch tensor of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    # Replace "pass" statement with your code
    scores, _ = nn_loss_part1(params, X)
    y_pred = scores.argmax(dim=1)
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

toy_X, toy_y, params = get_toy_data()

stats = nn_train(params, nn_loss_part2, nn_predict, toy_X, toy_y, toy_X, toy_y,
                 learning_rate=1e-1, reg=1e-6,
                 num_iters=200, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'], 'o')
plt.xlabel('Iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()

# Plot the loss function and train / validation accuracies
plt.plot(stats['train_acc_history'], 'o', label='train')
plt.plot(stats['val_acc_history'], 'o', label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.legend()
plt.show()

class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, device='cuda',
                 std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        # fix random seed before we generate a set of parameters
        coutils.utils.fix_random_seed()

        self.params = {}
        self.params['W1'] = std * torch.randn(input_size, hidden_size, device=device)
        self.params['b1'] = torch.zeros(hidden_size, device=device)
        self.params['W2'] = std * torch.randn(hidden_size, output_size, device=device)
        self.params['b2'] = torch.zeros(output_size, device=device)

    def _loss(self, params, X, y=None, reg=0.0):
        return nn_loss_part2(params, X, y, reg)

    def loss(self, X, y=None, reg=0.0):
        return self._loss(self.params, X, y, reg)

    def _train(self, params, loss_func, pred_func, X, y, X_val, y_val,
               learning_rate=1e-3, learning_rate_decay=0.95,
               reg=5e-6, num_iters=100,
               batch_size=200, verbose=False):
        return nn_train(params, loss_func, pred_func, X, y, X_val, y_val,
                        learning_rate, learning_rate_decay,
                        reg, num_iters, batch_size, verbose)

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        return self._train(self.params, self._loss, self._predict,
                           X, y, X_val, y_val,
                           learning_rate, learning_rate_decay,
                           reg, num_iters, batch_size, verbose)

    def _predict(self, params, loss_func, X):
        return nn_predict(params, loss_func, X)

    def predict(self, X):
        return self._predict(self.params, self._loss, X)

    @staticmethod
    def get_CIFAR10_data(validation_ratio=0.05):
        """
        Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
        it for the linear classifier. These are the same steps as we used for the
        SVM, but condensed to a single function.
        """
        X_train, y_train, X_test, y_test = coutils.data.cifar10()

        # load every data on cuda
        X_train = X_train.cuda()
        y_train = y_train.cuda()
        X_test = X_test.cuda()
        y_test = y_test.cuda()

        # 0. Visualize some examples from the dataset.
        class_names = [
            'plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        img = coutils.utils.visualize_dataset(X_train, y_train, 12, class_names)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        # 1. Normalize the data: subtract the mean RGB (zero mean)
        mean_image = X_train.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        X_train -= mean_image
        X_test -= mean_image

        # 2. Reshape the image data into rows
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        # 3. take the validation set from the training set
        # Note: It should not be taken from the test set
        # For random permumation, you can use torch.randperm or torch.randint
        # But, for this homework, we use slicing instead.
        num_training = int(X_train.shape[0] * (1.0 - validation_ratio))
        num_validation = X_train.shape[0] - num_training

        # return the dataset
        data_dict = {}
        data_dict['X_val'] = X_train[num_training:num_training + num_validation]
        data_dict['y_val'] = y_train[num_training:num_training + num_validation]
        data_dict['X_train'] = X_train[0:num_training]
        data_dict['y_train'] = y_train[0:num_training]

        data_dict['X_test'] = X_test
        data_dict['y_test'] = y_test
        return data_dict


# Invoke the above function to get our data.
data_dict = get_CIFAR10_data()
print('Train data shape: ', data_dict['X_train'].shape)
print('Train labels shape: ', data_dict['y_train'].shape)
print('Validation data shape: ', data_dict['X_val'].shape)
print('Validation labels shape: ', data_dict['y_val'].shape)
print('Test data shape: ', data_dict['X_test'].shape)
print('Test labels shape: ', data_dict['y_test'].shape)

input_size = 3 * 32 * 32
hidden_size = 36
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(data_dict['X_train'], data_dict['y_train'],
                  data_dict['X_val'], data_dict['y_val'],
                  num_iters=500, batch_size=1000,
                  learning_rate=1e-2, learning_rate_decay=0.95,
                  reg=0.25, verbose=True)

# Predict on the validation set
y_val_pred = net.predict(data_dict['X_val'])
val_acc = 100.0 * (y_val_pred == data_dict['y_val']).float().mean().item()
print('Validation accuracy: %.2f%%' % val_acc)

# Plot the loss function and train / validation accuracies
def plot_stats(stat_dict):
  plt.subplot(1, 2, 1)
  plt.plot(stat_dict['loss_history'], 'o')
  plt.title('Loss history')
  plt.xlabel('Iteration')
  plt.ylabel('Loss')

  plt.subplot(1, 2, 2)
  plt.plot(stat_dict['train_acc_history'], 'o-', label='train')
  plt.plot(stat_dict['val_acc_history'], 'o-', label='val')
  plt.title('Classification accuracy history')
  plt.xlabel('Epoch')
  plt.ylabel('Clasification accuracy')
  plt.legend()
  
  plt.gcf().set_size_inches(14, 4)
  plt.show()

plot_stats(stats)

def visualize_grid(Xs, ubound=255.0, padding=1):
  """
  Reshape a 4D tensor of image data to a grid for easy visualization.

  Inputs:
  - Xs: Data of shape (N, H, W, C)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  (N, H, W, C) = Xs.shape
  # print(Xs.shape)
  grid_size = int(math.ceil(math.sqrt(N)))
  grid_height = H * grid_size + padding * (grid_size - 1)
  grid_width = W * grid_size + padding * (grid_size - 1)
  grid = torch.zeros((grid_height, grid_width, C), device=Xs.device)
  next_idx = 0
  y0, y1 = 0, H
  for y in range(grid_size):
    x0, x1 = 0, W
    for x in range(grid_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = torch.min(img), torch.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        # grid[y0:y1, x0:x1] = Xs[next_idx]
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  # print(grid.shape)
  return grid


# Visualize the weights of the network
def show_net_weights(net):
  W1 = net.params['W1']
  W1 = W1.reshape(3, 32, 32, -1).transpose(0, 3)
  plt.imshow(visualize_grid(W1, padding=3).type(torch.uint8).cpu())
  plt.gca().axis('off')
  plt.show()

show_net_weights(net)

def plot_acc_curves(stat_dict):
  plt.subplot(1, 2, 1)
  for key, single_stats in stat_dict.items():
    plt.plot(single_stats['train_acc_history'], label=str(key))
  plt.title('Train accuracy history')
  plt.xlabel('Epoch')
  plt.ylabel('Clasification accuracy')

  plt.subplot(1, 2, 2)
  for key, single_stats in stat_dict.items():
    plt.plot(single_stats['val_acc_history'], label=str(key))
  plt.title('Validation accuracy history')
  plt.xlabel('Epoch')
  plt.ylabel('Clasification accuracy')
  plt.legend()

  plt.gcf().set_size_inches(14, 5)
  plt.show()

hidden_sizes = [2, 8, 32, 128, 512] 
lr = 0.1
reg = 0.001

stat_dict = {}
for hs in hidden_sizes:
  print('train with hidden size: {}'.format(hs))
  net = TwoLayerNet(3 * 32 * 32, hs, 10, device=data_dict['X_train'].device)
  stats = net.train(data_dict['X_train'], data_dict['y_train'], data_dict['X_val'], data_dict['y_val'],
            num_iters=3000, batch_size=1000,
            learning_rate=lr, learning_rate_decay=0.95,
            reg=reg, verbose=False)
  stat_dict[hs] = stats

plot_acc_curves(stat_dict)

hs = 128
lr = 1.0
regs = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

stat_dict = {}
for reg in regs:
  print('train with regularization: {}'.format(reg))
  net = TwoLayerNet(3 * 32 * 32, hs, 10, device=data_dict['X_train'].device)
  stats = net.train(data_dict['X_train'], data_dict['y_train'], data_dict['X_val'], data_dict['y_val'],
            num_iters=3000, batch_size=1000,
            learning_rate=lr, learning_rate_decay=0.95,
            reg=reg, verbose=False)
  stat_dict[reg] = stats

plot_acc_curves(stat_dict)

hs = 128
lrs = [1e-3, 1e-2, 1e-1, 1e0, 1e1]
reg = 1e-4

stat_dict = {}
for lr in lrs:
  print('train with learning rate: {}'.format(lr))
  net = TwoLayerNet(3 * 32 * 32, hs, 10, device=data_dict['X_train'].device)
  stats = net.train(data_dict['X_train'], data_dict['y_train'], data_dict['X_val'], data_dict['y_val'],
            num_iters=3000, batch_size=1000,
            learning_rate=lr, learning_rate_decay=0.95,
            reg=reg, verbose=False)
  stat_dict[lr] = stats

plot_acc_curves(stat_dict)

best_net = None # store the best model into this 

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
# Replace "pass" statement with your code
best_val = -1
best_params = None

learning_rates = [1e-3, 1e-2, 1e-1]
hidden_sizes = [50, 100, 200]
regularization_strengths = [1e-4, 1e-3, 1e-2]
num_iters = 3000

for lr in learning_rates:
    for hidden_size in hidden_sizes:
        for reg in regularization_strengths:
            params = {
                'W1': torch.randn(data_dict['X_train'].shape[1], hidden_size) * 0.001,
                'b1': torch.zeros(hidden_size),
                'W2': torch.randn(hidden_size, 10) * 0.001,
                'b2': torch.zeros(10),
            }

            stats = nn_train(
                params, nn_loss_part2, nn_predict,
                data_dict['X_train'], data_dict['y_train'],
                data_dict['X_val'], data_dict['y_val'],
                learning_rate=lr, reg=reg, num_iters=num_iters,
                batch_size=200, verbose=False
            )

            val_acc = stats['val_acc_history'][-1]
            if val_acc > best_val:
                best_val = val_acc
                best_net = params

#################################################################################
#                               END OF YOUR CODE                                #
#################################################################################

# Check the validation-set accuracy of your best model
y_val_preds = best_net.predict(data_dict['X_val'])
val_acc = 100 * (y_val_preds == data_dict['y_val']).float().mean().item()
print('Best val-set accuracy: %.2f%%' % val_acc)

# visualize the weights of the best network
show_net_weights(best_net)

y_test_preds = best_net.predict(data_dict['X_test'])
test_acc = 100 * (y_test_preds == data_dict['y_test']).float().mean().item()
print('Test accuracy: %.2f%%' % test_acc)