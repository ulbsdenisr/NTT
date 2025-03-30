import torch
import coutils
import torchvision
import matplotlib.pyplot as plt
import statistics
def compute_distances_two_loops(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation uses a naive set of nested loops over the training and
  test data.
  
  Inputs:
  - x_train: Torch tensor of shape (num_train, C, H, W)
  - x_test: Torch tensor of shape (num_test, C, H, W)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]
  dists = x_train.new_zeros(num_train, num_test)
  ##############################################################################
  # TODO: Implement this function using a pair of nested loops over the        #
  # training data and the test data.                                           #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  ##############################################################################
  # Replace "pass" statement with your code
  for i in range(num_train):
    for j in range(num_test):
      diff=x_train[i].flatten() - x_test[j].flatten()
      dists[i,j]=torch.sum(diff**2)
  return dists
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
 
num_train = 500
num_test = 250
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

x_train = torch.stack([data[0] for data in trainset][:num_train])
y_train = torch.tensor([data[1] for data in trainset][:num_train])
x_test = torch.stack([data[0] for data in testset][:num_test])
y_test = torch.tensor([data[1] for data in testset][:num_test])


dists = compute_distances_two_loops(x_train, x_test)
print('dists has shape: ', dists.shape)

plt.imshow(dists.numpy(), cmap='gray', interpolation='none')
plt.colorbar()
plt.xlabel('test')
plt.ylabel('train')
plt.show()
#///////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\

def compute_distances_one_loop(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation uses only a single loop over the training data.

  Inputs:
  - x_train: Torch tensor of shape (num_train, C, H, W)
  - x_test: Torch tensor of shape (num_test, C, H, W)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]
  dists = x_train.new_zeros(num_train, num_test)
  ##############################################################################
  # TODO: Implement this function using only a single loop over x_train.       #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  ##############################################################################
  # Replace "pass" statement with your code
  for i in range(num_train):
        diff = x_train[i].flatten() - x_test.view(num_test, -1)  
        dists[i, :] = torch.sum(diff ** 2, dim=1)
  return dists
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################


torch.manual_seed(0)
x_train_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)
x_test_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)

dists_one = compute_distances_one_loop(x_train_rand, x_test_rand)
dists_two = compute_distances_two_loops(x_train_rand, x_test_rand)
difference = (dists_one - dists_two).pow(2).sum().sqrt().item()
print('Difference: ', difference)
if difference < 1e-4:
  print('Good! The distance matrices match')
else:
  print('Uh-oh! The distance matrices are different')

  #///////////////////////\\\\\\\\\\\\\\\\\\\\\\

def compute_distances_no_loops(x_train, x_test):
  """
  Computes the squared Euclidean distance between each element of the training
  set and each element of the test set. Images should be flattened and treated
  as vectors.

  This implementation should not use any Python loops. For memory-efficiency,
  it also should not create any large intermediate tensors; in particular you
  should not create any intermediate tensors with O(num_train*num_test)
  elements.

  Inputs:
  - x_train: Torch tensor of shape (num_train, C, H, W)
  - x_test: Torch tensor of shape (num_test, C, H, W)

  Returns:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  """
  # Initialize dists to be a tensor of shape (num_train, num_test) with the
  # same datatype and device as x_train
  num_train = x_train.shape[0]
  num_test = x_test.shape[0]
  dists = x_train.new_zeros(num_train, num_test)
  ##############################################################################
  # TODO: Implement this function without using any explicit loops and without #
  # creating any intermediate tensors with O(num_train * num_test) elements.   #
  #                                                                            #
  # You may not use torch.norm (or its instance method variant), nor any       #
  # functions from torch.nn or torch.nn.functional.                            #
  #                                                                            #
  # HINT: Try to formulate the Euclidean distance using two broadcast sums     #
  #       and a matrix multiply.                                               #
  ##############################################################################
  # Replace "pass" statement with your code
  x_train_flat = x_train.view(num_train, -1)
  x_test_flat = x_test.view(num_test, -1)

  x_train_sq = torch.sum(x_train_flat ** 2, dim=1, keepdim=True)
  x_test_sq = torch.sum(x_test_flat ** 2, dim=1)
  cross_term = torch.mm(x_train_flat, x_test_flat.T)

  dists = x_train_sq - 2 * cross_term + x_test_sq
  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################
  return dists
torch.manual_seed(0)
x_train_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)
x_test_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)

dists_two = compute_distances_two_loops(x_train_rand, x_test_rand)
dists_none = compute_distances_no_loops(x_train_rand, x_test_rand)
difference = (dists_two - dists_none).pow(2).sum().sqrt().item()
print('Difference: ', difference)
if difference < 1e-4:
  print('Good! The distance matrices match')
else:
  print('Uh-oh! The distance matrices are different')

  #//////////////////////////\\\\\\\\\\\\\\\\\\\\\\

  import time

def timeit(f, *args):
  tic = time.time()
  f(*args) 
  toc = time.time()
  return toc - tic

torch.manual_seed(0)
x_train_rand = torch.randn(500, 3, 32, 32)
x_test_rand = torch.randn(500, 3, 32, 32)

two_loop_time = timeit(compute_distances_two_loops, x_train_rand, x_test_rand)
print('Two loop version took %.2f seconds' % two_loop_time)

one_loop_time = timeit(compute_distances_one_loop, x_train_rand, x_test_rand)
speedup = two_loop_time / one_loop_time
print('One loop version took %.2f seconds (%.1fX speedup)'
      % (one_loop_time, speedup))

no_loop_time = timeit(compute_distances_no_loops, x_train_rand, x_test_rand)
speedup = two_loop_time / no_loop_time
print('No loop version took %.2f seconds (%.1fX speedup)'
      % (no_loop_time, speedup))

def predict_labels(dists, y_train, k=1):
  """
  Given distances between all pairs of training and test samples, predict a
  label for each test sample by taking a majority vote among its k nearest
  neighbors in the training set.
 
  In the event of a tie, this function should return the smaller label. For
  example, if k=5 and the 5 nearest neighbors to a test example have labels
  [1, 2, 1, 2, 3] then there is a tie between 1 and 2 (each have 2 votes), so
  we should return 1 since it is the smaller label.

  Inputs:
  - dists: Torch tensor of shape (num_train, num_test) where dists[i, j] is the
    squared Euclidean distance between the ith training point and the jth test
    point.
  - y_train: Torch tensor of shape (y_train,) giving labels for all training
    samples. Each label is an integer in the range [0, num_classes - 1]
  - k: The number of nearest neighbors to use for classification.

  Returns:
  - y_pred: A torch int64 tensor of shape (num_test,) giving predicted labels
    for the test data, where y_pred[j] is the predicted label for the jth test
    example. Each label should be an integer in the range [0, num_classes - 1].
  """
  num_train, num_test = dists.shape
  y_pred = torch.zeros(num_test, dtype=torch.int64)
  ##############################################################################
  # TODO: Implement this function. You may use an explicit loop over the test  #
  # samples. Hint: Look up the function torch.topk                             #
  ##############################################################################
  # Replace "pass" statement with your code
  import torch

def predict_labels(dists, y_train, k=1):
    num_train, num_test = dists.shape
    y_pred = torch.zeros(num_test, dtype=torch.int64)

    for j in range(num_test):
        # Găsim indicele celor mai apropiați k vecini
        _, indices = torch.topk(dists[:, j], k, largest=False)  # Sortăm ascendent după distanță

        # Selectăm etichetele celor k vecini
        nearest_labels = y_train[indices]

        # Determinăm eticheta majoritară
        unique_labels, counts = torch.unique(nearest_labels, return_counts=True)
        max_count = torch.max(counts)
        best_labels = unique_labels[counts == max_count]

        # Alegem cel mai mic label în caz de egalitate
        y_pred[j] = torch.min(best_labels)

    return y_pred

  ##############################################################################
  #                             END OF YOUR CODE                               #
  ##############################################################################

class KnnClassifier:
  def __init__(self, x_train, y_train):
    """
    Create a new K-Nearest Neighbor classifier with the specified training data.
    In the initializer we simply memorize the provided training data.

    Inputs:
    - x_train: Torch tensor of shape (num_train, C, H, W) giving training data
    - y_train: int64 torch tensor of shape (num_train,) giving training labels
    """
    self.x_train = x_train.contiguous()
    self.y_train = y_train.contiguous()
  
  def predict(self, x_test, k=1):
    """
    Make predictions using the classifier.
   
    Inputs:
    - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
    - k: The number of neighbors to use for predictions
  
    Returns:
    - y_test_pred: Torch tensor of shape (num_test,) giving predicted labels
      for the test samples.
    """
    dists = compute_distances_no_loops(self.x_train, x_test.contiguous())
    y_test_pred = predict_labels(dists, self.y_train, k=k)
    return y_test_pred
  
  def check_accuracy(self, x_test, y_test, k=1, quiet=False):
    """
    Utility method for checking the accuracy of this classifier on test data.
    Returns the accuracy of the classifier on the test data, and also prints a
    message giving the accuracy.

    Inputs:
    - x_test: Torch tensor of shape (num_test, C, H, W) giving test samples
    - y_test: int64 torch tensor of shape (num_test,) giving test labels
    - k: The number of neighbors to use for prediction
    - quiet: If True, don't print a message.
  
    Returns:
    - accuracy: Accuracy of this classifier on the test data, as a percent.
      Python float in the range [0, 100]
    """
    y_test_pred = self.predict(x_test, k=k)
    num_samples = x_test.shape[0]
    num_correct = (y_test == y_test_pred).sum().item()
    accuracy = 100.0 * num_correct / num_samples
    msg = (f'Got {num_correct} / {num_samples} correct; '
           f'accuracy is {accuracy:.2f}%')
    if not quiet:
      print(msg)
    return accuracy
  
  num_train = 5000
num_test = 500
x_train, y_train, x_test, y_test = coutils.data.cifar10(num_train, num_test)

classifier = KnnClassifier(x_train, y_train)
classifier.check_accuracy(x_test, y_test, k=1)

classifier.check_accuracy(x_test, y_test, k=5)


def knn_cross_validate(x_train, y_train, num_folds=5, k_choices=None):
  """
  Perform cross-validation for KnnClassifier.

  Inputs:
  - x_train: Tensor of shape (num_train, C, H, W) giving all training data
  - y_train: int64 tensor of shape (num_train,) giving labels for training data
  - num_folds: Integer giving the number of folds to use
  - k_choices: List of integers giving the values of k to try
 
  Returns:
  - k_to_accuracies: Dictionary mapping values of k to lists, where
    k_to_accuracies[k][i] is the accuracy on the ith fold of a KnnClassifier
    that uses k nearest neighbors.
  """
  if k_choices is None:
    # Use default values
    k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

  # First we divide the training data into num_folds equally-sized folds.
  x_train_folds = []
  y_train_folds = []
  ##############################################################################
  # TODO: Split the training data and images into folds. After splitting,      #
  # x_train_folds and y_train_folds should be lists of length num_folds, where #
  # y_train_folds[i] is the label vector for images in x_train_folds[i].       #
  # Hint: torch.chunk                                                          #
  ##############################################################################
  # Replace "pass" statement with your code

def knn_cross_validate(x_train, y_train, num_folds=5, k_choices=None):
    if k_choices is None:
        k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    
    x_train_folds = list(torch.chunk(x_train, num_folds, dim=0))
    y_train_folds = list(torch.chunk(y_train, num_folds, dim=0))

    return x_train_folds, y_train_folds  

  ##############################################################################
  #                            END OF YOUR CODE                                #
  ##############################################################################
  
  # A dictionary holding the accuracies for different values of k that we find
  # when running cross-validation. After running cross-validation,
  # k_to_accuracies[k] should be a list of length num_folds giving the different
  # accuracies we found when trying KnnClassifiers that use k neighbors.
k_to_accuracies = {}

  ##############################################################################
  # TODO: Perform cross-validation to find the best value of k. For each value #
  # of k in k_choices, run the k-nearest-neighbor algorithm num_folds times;   #
  # in each case you'll use all but one fold as training data, and use the     #
  # last fold as a validation set. Store the accuracies for all folds and all  #
  # values in k in k_to_accuracies.   HINT: torch.cat                          #
  ##############################################################################
  # Replace "pass" statement with your code
import torch

def knn_cross_validate(x_train, y_train, num_folds=5, k_choices=None):
    if k_choices is None:
        k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    x_train_folds = list(torch.chunk(x_train, num_folds, dim=0))
    y_train_folds = list(torch.chunk(y_train, num_folds, dim=0))

    k_to_accuracies = {k: [] for k in k_choices}

    for fold in range(num_folds):
        x_val = x_train_folds[fold]
        y_val = y_train_folds[fold]

        x_train_cv = torch.cat([x_train_folds[i] for i in range(num_folds) if i != fold], dim=0)
        y_train_cv = torch.cat([y_train_folds[i] for i in range(num_folds) if i != fold], dim=0)

        dists = compute_distances_no_loops(x_train_cv, x_val)

        for k in k_choices:
            y_pred = predict_labels(dists, y_train_cv, k=k)
            accuracy = torch.mean((y_pred == y_val).float()).item()  # Calculăm acuratețea
            k_to_accuracies[k].append(accuracy)

    return k_to_accuracies

  ##############################################################################
  #                            END OF YOUR CODE                                #
  ##############################################################################

best_k = 1
##############################################################################
# TODO: Use the results of cross-validation stored in k_to_accuracies to     #
# choose the value of k, and store the result in best_k. You should choose   #
# the value of k that has the highest mean accuracy accross all folds.       #
##############################################################################
# Replace "pass" statement with your code
best_accuracy = 0

for k in k_to_accuracies:
    mean_accuracy = torch.mean(torch.tensor(k_to_accuracies[k])).item()
    
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_k = k

print('Best k is', best_k)

##############################################################################
#                            END OF YOUR CODE                                #
##############################################################################
    
print('Best k is ', best_k)
classifier = KnnClassifier(x_train, y_train)
classifier.check_accuracy(x_test, y_test, k=best_k)


x_train_all, y_train_all, x_test_all, y_test_all = coutils.data.cifar10()
classifier = KnnClassifier(x_train_all, y_train_all)
classifier.check_accuracy(x_test_all, y_test_all, k=best_k)