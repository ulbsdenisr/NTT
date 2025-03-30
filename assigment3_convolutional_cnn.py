import math
import torch
import coutils
from coutils import extract_drive_file_id, register_colab_notebooks, \
                    fix_random_seed, rel_error, compute_numeric_gradient, Solver
import matplotlib.pyplot as plt
import time


# for plotting
plt.ion()
plt.rcParams['figure.figsize'] = (10.0, 8.0) 
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# data type and device for torch.tensor
to_float = {'dtype': torch.float, 'device': 'cpu'}
to_float_cuda = {'dtype': torch.float, 'device': 'cuda'}
to_double = {'dtype': torch.double, 'device': 'cpu'}
to_double_cuda = {'dtype': torch.double, 'device': 'cuda'}
to_long = {'dtype': torch.long, 'device': 'cpu'}
to_long_cuda = {'dtype': torch.long, 'device': 'cuda'}

FULLY_CONNECTED_NOTEBOOK_LINK = ""

fcn_id = extract_drive_file_id(FULLY_CONNECTED_NOTEBOOK_LINK)
print('Google Drive file id: "%s"' % fcn_id)
register_colab_notebooks({'assigment3_full_cnn': fcn_id})

from assigment3_full_cnn import get_CIFAR10_data
from assigment3_full_cnn import Linear, ReLU, Linear_ReLU, Dropout
from assigment3_full_cnn import svm_loss, softmax_loss
from assigment3_full_cnn import sgd_momentum, rmsprop, adam

print('Import successful!')

# Invoke the above function to get our data.
data_dict = get_CIFAR10_data(visualize=True)
print('Train data shape: ', data_dict['X_train'].shape)
print('Train labels shape: ', data_dict['y_train'].shape)
print('Validation data shape: ', data_dict['X_val'].shape)
print('Validation labels shape: ', data_dict['y_val'].shape)
print('Test data shape: ', data_dict['X_test'].shape)
print('Test labels shape: ', data_dict['y_test'].shape)


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each filter
        spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
            - 'stride': The number of pixels between adjacent receptive fields in the
              horizontal and vertical directions.
            - 'pad': The number of pixels that will be used to zero-pad the input. 

        During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
        along the height and width axes of the input. Be careful not to modify the original
        input x directly.

        Returns a tuple of:
        - out: Output data, of shape (N, F, H', W') where H' and W' are given by
            H' = 1 + (H + 2 * pad - HH) / stride
            W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        import torch.nn.functional as F

        
        N, C, H, W = x.shape
        F_, C, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']

    
        H_out = 1 + (H + 2 * pad - HH) // stride
        W_out = 1 + (W + 2 * pad - WW) // stride

        x_padded = F.pad(x, (pad, pad, pad, pad))

        out = torch.zeros((N, F_, H_out, W_out), device=x.device, dtype=x.dtype)

        for n in range(N):
            for f in range(F_):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * stride
                        h_end = h_start + HH
                        w_start = j * stride
                        w_end = w_start + WW

                        out[n, f, i, j] = torch.sum(
                            x_padded[n, :, h_start:h_end, w_start:w_end] * w[f]
                        ) + b[f]

        cache = (x, w, b, conv_param)
        return out, cache


Conv.forward = Conv.forward

x_shape = torch.tensor((2, 3, 4, 4))
w_shape = torch.tensor((3, 3, 4, 4))
x = torch.linspace(-0.1, 0.5, steps=torch.prod(x_shape), **to_double_cuda).reshape(*x_shape)
w = torch.linspace(-0.2, 0.3, steps=torch.prod(w_shape), **to_double_cuda).reshape(*w_shape)
b = torch.linspace(-0.1, 0.2, steps=3, **to_double_cuda)

conv_param = {'stride': 2, 'pad': 1}
out, _ = Conv.forward(x, w, b, conv_param)
correct_out = torch.tensor([[[[-0.08759809, -0.10987781],
                              [-0.18387192, -0.2109216 ]],
                             [[ 0.21027089,  0.21661097],
                              [ 0.22847626,  0.23004637]],
                             [[ 0.50813986,  0.54309974],
                              [ 0.64082444,  0.67101435]]],
                            [[[-0.98053589, -1.03143541],
                              [-1.19128892, -1.24695841]],
                             [[ 0.69108355,  0.66880383],
                              [ 0.59480972,  0.56776003]],
                             [[ 2.36270298,  2.36904306],
                              [ 2.38090835,  2.38247847]]]], **to_double_cuda)

# Compare your output to ours; difference should be around e-8
print('Testing Conv.forward')
print('difference: ', rel_error(out, correct_out))

from imageio import imread
from PIL import Image
from torchvision.transforms import ToTensor

kitten_url = 'https://web.eecs.umich.edu/~justincj/teaching/eecs498/assets/a3/kitten.jpg'
puppy_url = 'https://web.eecs.umich.edu/~justincj/teaching/eecs498/assets/a3/puppy.jpg'

kitten = imread(kitten_url)
puppy = imread(puppy_url)
# kitten is wide, and puppy is already square
d = kitten.shape[1] - kitten.shape[0]
kitten_cropped = kitten[:, d//2:-d//2, :]

img_size = 200   # Make this smaller if it runs too slow
resized_puppy = ToTensor()(Image.fromarray(puppy).resize((img_size, img_size)))
resized_kitten = ToTensor()(Image.fromarray(kitten_cropped).resize((img_size, img_size)))
x = torch.stack([resized_puppy, resized_kitten])

# Set up a convolutional weights holding 2 filters, each 3x3
w = torch.zeros(2, 3, 3, 3, dtype=x.dtype)

# The first filter converts the image to grayscale.
# Set up the red, green, and blue channels of the filter.
w[0, 0, :, :] = torch.tensor([[0, 0, 0], [0, 0.3, 0], [0, 0, 0]])
w[0, 1, :, :] = torch.tensor([[0, 0, 0], [0, 0.6, 0], [0, 0, 0]])
w[0, 2, :, :] = torch.tensor([[0, 0, 0], [0, 0.1, 0], [0, 0, 0]])

# Second filter detects horizontal edges in the blue channel.
w[1, 2, :, :] = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# Vector of biases. We don't need any bias for the grayscale
# filter, but for the edge detection filter we want to add 128
# to each output so that nothing is negative.
b = torch.tensor([0, 128], dtype=x.dtype)

# Compute the result of convolving each input in x with each filter in w,
# offsetting by b, and storing the results in out.
out, _ = Conv.forward(x, w, b, {'stride': 1, 'pad': 1})

def imshow_no_ax(img, normalize=True):
  """ Tiny helper to show images as uint8 and remove axis labels """
  if normalize:
    img_max, img_min = img.max(), img.min()
    img = 255.0 * (img - img_min) / (img_max - img_min)
  plt.imshow(img)
  plt.gca().axis('off')

# Show the original images and the results of the conv operation
plt.subplot(2, 3, 1)
imshow_no_ax(puppy, normalize=False)
plt.title('Original image')
plt.subplot(2, 3, 2)
imshow_no_ax(out[0, 0])
plt.title('Grayscale')
plt.subplot(2, 3, 3)
imshow_no_ax(out[0, 1])
plt.title('Edges')
plt.subplot(2, 3, 4)
imshow_no_ax(kitten_cropped, normalize=False)
plt.subplot(2, 3, 5)
imshow_no_ax(out[1, 0])
plt.subplot(2, 3, 6)
imshow_no_ax(out[1, 1])
plt.show()

def conv_backward(dout, cache):
    x, w, b, conv_param = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    N, C, H, W = x.shape
    F_, C, HH, WW = w.shape
    H_out, W_out = dout.shape[2], dout.shape[3]

    x_padded = torch.nn.functional.pad(x, (pad, pad, pad, pad))
    dx_padded = torch.zeros_like(x_padded)
    dw = torch.zeros_like(w)
    db = dout.sum(dim=(0, 2, 3))

    for n in range(N):
        for f in range(F_):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW

                    dw[f] += dout[n, f, i, j] * x_padded[n, :, h_start:h_end, w_start:w_end]
                    dx_padded[n, :, h_start:h_end, w_start:w_end] += dout[n, f, i, j] * w[f]

    dx = dx_padded[:, :, pad:-pad, pad:-pad] if pad > 0 else dx_padded

    return dx, dw, db

Conv.backward = conv_backward

fix_random_seed(0)
x = torch.randn(4, 3, 5, 5, **to_double_cuda)
w = torch.randn(2, 3, 3, 3, **to_double_cuda)
b = torch.randn(2, **to_double_cuda)
dout = torch.randn(4, 2, 5, 5, **to_double_cuda)
conv_param = {'stride': 1, 'pad': 1}

dx_num = compute_numeric_gradient(lambda x: Conv.forward(x, w, b, conv_param)[0], x, dout)
dw_num = compute_numeric_gradient(lambda w: Conv.forward(x, w, b, conv_param)[0], w, dout)
db_num = compute_numeric_gradient(lambda b: Conv.forward(x, w, b, conv_param)[0], b, dout)

out, cache = Conv.forward(x, w, b, conv_param)
dx, dw, db = Conv.backward(dout, cache)

print('Testing Conv.backward function')
print('dx error: ', rel_error(dx, dx_num))
print('dw error: ', rel_error(dw, dw_num))
print('db error: ', rel_error(db, db_num))

class MaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    raise NotImplementedError

  @staticmethod
  def backward(dout, cache):
    raise NotImplementedError
  
def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here.

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max-pooling forward pass                              #
    #############################################################################
    # Replace "pass" statement with your code
    pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    N, C, H, W = x.shape
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    out = torch.zeros((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    for n in range(N):
        for c in range(C):
            for i in range(H_out):
                for j in range(W_out):
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width
                    
                    out[n, c, i, j] = torch.max(x[n, c, h_start:h_end, w_start:w_end])
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


MaxPool.forward = max_pool_forward

x_shape = torch.tensor((2, 3, 4, 4))
x = torch.linspace(-0.3, 0.4, steps=torch.prod(x_shape), **to_double_cuda).reshape(*x_shape)
pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}

out, _ = MaxPool.forward(x, pool_param)

correct_out = torch.tensor([[[[-0.26315789, -0.24842105],
                              [-0.20421053, -0.18947368]],
                             [[-0.14526316, -0.13052632],
                              [-0.08631579, -0.07157895]],
                             [[-0.02736842, -0.01263158],
                              [ 0.03157895,  0.04631579]]],
                            [[[ 0.09052632,  0.10526316],
                              [ 0.14947368,  0.16421053]],
                             [[ 0.20842105,  0.22315789],
                              [ 0.26736842,  0.28210526]],
                             [[ 0.32631579,  0.34105263],
                              [ 0.38526316,  0.4       ]]]], **to_double_cuda)

# Compare your output with ours. Difference should be on the order of e-8.
print('Testing MaxPool.forward function:')
print('difference: ', rel_error(out, correct_out))

fix_random_seed(0)
x = torch.randn(3, 2, 8, 8, **to_double_cuda)
dout = torch.randn(3, 2, 4, 4, **to_double_cuda)
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

dx_num = compute_numeric_gradient(lambda x: MaxPool.forward(x, pool_param)[0], x, dout)

out, cache = MaxPool.forward(x, pool_param)
dx = MaxPool.backward(dout, cache)

print('Testing MaxPool.backward function:')
print('dx error: ', rel_error(dx, dx_num))

class FastConv(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']
    layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
    layer.weight = torch.nn.Parameter(w)
    layer.bias = torch.nn.Parameter(b)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, w, b, conv_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, _, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
      dw = layer.weight.grad.detach()
      db = layer.bias.grad.detach()
      layer.weight.grad = layer.bias.grad = None
    except RuntimeError:
      dx, dw, db = torch.zeros_like(tx), torch.zeros_like(layer.weight), torch.zeros_like(layer.bias)
    return dx, dw, db


class FastMaxPool(object):

  @staticmethod
  def forward(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width), stride=stride)
    tx = x.detach()
    tx.requires_grad = True
    out = layer(tx)
    cache = (x, pool_param, tx, out, layer)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    try:
      x, _, tx, out, layer = cache
      out.backward(dout)
      dx = tx.grad.detach()
    except RuntimeError:
      dx = torch.zeros_like(tx)
    return dx
  
  # Rel errors should be around e-11 or less
fix_random_seed(0)
x = torch.randn(10, 3, 31, 31, **to_double)
w = torch.randn(25, 3, 3, 3, **to_double)
b = torch.randn(25, **to_double)
dout = torch.randn(10, 25, 16, 16, **to_double)
x_cuda, w_cuda, b_cuda, dout_cuda = x.to('cuda'), w.to('cuda'), b.to('cuda'), dout.to('cuda')
conv_param = {'stride': 2, 'pad': 1}

t0 = time.time()
out_naive, cache_naive = Conv.forward(x, w, b, conv_param)
t1 = time.time()
out_fast, cache_fast = FastConv.forward(x, w, b, conv_param)
t2 = time.time()
out_fast_cuda, cache_fast_cuda = FastConv.forward(x_cuda, w_cuda, b_cuda, conv_param)
t3 = time.time()

print('Testing FastConv.forward:')
print('Naive: %fs' % (t1 - t0))
print('Fast: %fs' % (t2 - t1))
print('Fast CUDA: %fs' % (t3 - t2))
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('Speedup CUDA: %fx' % ((t1 - t0) / (t3 - t2)))
print('Difference: ', rel_error(out_naive, out_fast))
print('Difference CUDA: ', rel_error(out_naive, out_fast_cuda.to(out_naive.device)))

t0 = time.time()
dx_naive, dw_naive, db_naive = Conv.backward(dout, cache_naive)
t1 = time.time()
dx_fast, dw_fast, db_fast = FastConv.backward(dout, cache_fast)
t2 = time.time()
dx_fast_cuda, dw_fast_cuda, db_fast_cuda = FastConv.backward(dout_cuda, cache_fast_cuda)
t3 = time.time()

print('\nTesting FastConv.backward:')
print('Naive: %fs' % (t1 - t0))
print('Fast: %fs' % (t2 - t1))
print('Fast CUDA: %fs' % (t3 - t2))
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('Speedup CUDA: %fx' % ((t1 - t0) / (t3 - t2)))
print('dx difference: ', rel_error(dx_naive, dx_fast))
print('dw difference: ', rel_error(dw_naive, dw_fast))
print('db difference: ', rel_error(db_naive, db_fast))
print('dx difference CUDA: ', rel_error(dx_naive, dx_fast_cuda.to(dx_naive.device)))
print('dw difference CUDA: ', rel_error(dw_naive, dw_fast_cuda.to(dw_naive.device)))
print('db difference CUDA: ', rel_error(db_naive, db_fast_cuda.to(db_naive.device)))

# Relative errors should be close to 0.0
fix_random_seed(0)
x = torch.randn(40, 3, 32, 32, **to_double)
dout = torch.randn(40, 3, 16, 16, **to_double)
x_cuda, dout_cuda = x.to('cuda'), dout.to('cuda')
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

t0 = time.time()
out_naive, cache_naive = MaxPool.forward(x, pool_param)
t1 = time.time()
out_fast, cache_fast = FastMaxPool.forward(x, pool_param)
t2 = time.time()
out_fast_cuda, cache_fast_cuda = FastMaxPool.forward(x_cuda, pool_param)
t3 = time.time()

print('Testing FastMaxPool.forward:')
print('Naive: %fs' % (t1 - t0))
print('Fast: %fs' % (t2 - t1))
print('Fast CUDA: %fs' % (t3 - t2))
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('Speedup CUDA: %fx' % ((t1 - t0) / (t3 - t2)))
print('Difference: ', rel_error(out_naive, out_fast))
print('Difference CUDA: ', rel_error(out_naive, out_fast_cuda.to(out_naive.device)))

t0 = time.time()
dx_naive = MaxPool.backward(dout, cache_naive)
t1 = time.time()
dx_fast = FastMaxPool.backward(dout, cache_fast)
t2 = time.time()
dx_fast_cuda = FastMaxPool.backward(dout_cuda, cache_fast_cuda)
t3 = time.time()

print('\nTesting FastMaxPool.backward:')
print('Naive: %fs' % (t1 - t0))
print('Fast: %fs' % (t2 - t1))
print('Fast CUDA: %fs' % (t3 - t2))
print('Speedup: %fx' % ((t1 - t0) / (t2 - t1)))
print('Speedup CUDA: %fx' % ((t1 - t0) / (t3 - t2)))
print('dx difference: ', rel_error(dx_naive, dx_fast))
print('dx difference CUDA: ', rel_error(dx_naive, dx_fast_cuda.to(dx_naive.device)))

class Conv_ReLU(object):

  @staticmethod
  def forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    out, relu_cache = ReLU.forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = ReLU.backward(dout, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db


class Conv_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, conv_param, pool_param):
    """
    A convenience layer that performs a convolution, a ReLU, and a pool.
    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer
    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    s, relu_cache = ReLU.forward(a)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    da = ReLU.backward(ds, relu_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db
  
  fix_random_seed(0)
x = torch.randn(2, 3, 16, 16, **to_double_cuda)
w = torch.randn(3, 3, 3, 3, **to_double_cuda)
b = torch.randn(3, **to_double_cuda)
dout = torch.randn(2, 3, 8, 8, **to_double_cuda)
conv_param = {'stride': 1, 'pad': 1}
pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

out, cache = Conv_ReLU_Pool.forward(x, w, b, conv_param, pool_param)
dx, dw, db = Conv_ReLU_Pool.backward(dout, cache)

dx_num = compute_numeric_gradient(lambda x: Conv_ReLU_Pool.forward(x, w, b, conv_param, pool_param)[0], x, dout)
dw_num = compute_numeric_gradient(lambda w: Conv_ReLU_Pool.forward(x, w, b, conv_param, pool_param)[0], w, dout)
db_num = compute_numeric_gradient(lambda b: Conv_ReLU_Pool.forward(x, w, b, conv_param, pool_param)[0], b, dout)

# Relative errors should be around e-8 or less
print('Testing Conv_ReLU_Pool')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

fix_random_seed(0)
x = torch.randn(2, 3, 8, 8, **to_double_cuda)
w = torch.randn(3, 3, 3, 3, **to_double_cuda)
b = torch.randn(3, **to_double_cuda)
dout = torch.randn(2, 3, 8, 8, **to_double_cuda)
conv_param = {'stride': 1, 'pad': 1}

out, cache = Conv_ReLU.forward(x, w, b, conv_param)
dx, dw, db = Conv_ReLU.backward(dout, cache)

dx_num = compute_numeric_gradient(lambda x: Conv_ReLU.forward(x, w, b, conv_param)[0], x, dout)
dw_num = compute_numeric_gradient(lambda w: Conv_ReLU.forward(x, w, b, conv_param)[0], w, dout)
db_num = compute_numeric_gradient(lambda b: Conv_ReLU.forward(x, w, b, conv_param)[0], b, dout)

# Relative errors should be around e-8 or less
print('Testing Conv_ReLU:')
print('dx error: ', rel_error(dx_num, dx))
print('dw error: ', rel_error(dw_num, dw))
print('db error: ', rel_error(db_num, db))

class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  conv - relu - 2x2 max pool - linear - relu - linear - softmax
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dims=(3, 32, 32), num_filters=32, filter_size=7,
         hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
         dtype=torch.float, device='cpu'):
    """
    Initialize a new network.
    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Width/height of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian centered at 0.0   #
    # with standard deviation equal to weight_scale; biases should be          #
    # initialized to zero. All weights and biases should be stored in the      #
    #  dictionary self.params. Store weights and biases for the convolutional  #
    # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
    # weights and biases of the hidden linear layer, and keys 'W3' and 'b3'    #
    # for the weights and biases of the output linear layer.                   #
    #                                                                          #
    # IMPORTANT: For this assignment, you can assume that the padding          #
    # and stride of the first convolutional layer are chosen so that           #
    # **the width and height of the input are preserved**. Take a look at      #
    # the start of the loss() function to see how that happens.                #               
    ############################################################################
    # Replace "pass" statement with your code
    C, H, W = input_dims
    F = num_filters
    HH = filter_size
    WW = filter_size

    self.params['W1'] = torch.randn(F, C, HH, WW, dtype=dtype, device=device) * weight_scale
    self.params['b1'] = torch.zeros(F, dtype=dtype, device=device)

    pool_output_dim = (H // 2) * (W // 2) * F

    self.params['W2'] = torch.randn(pool_output_dim, hidden_dim, dtype=dtype, device=device) * weight_scale
    self.params['b2'] = torch.zeros(hidden_dim, dtype=dtype, device=device)

    self.params['W3'] = torch.randn(hidden_dim, num_classes, dtype=dtype, device=device) * weight_scale
    self.params['b3'] = torch.zeros(num_classes, dtype=dtype, device=device)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    Input / output: Same API as TwoLayerNet.
    """
    X = X.to(self.dtype)
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    #                                                                          #
    # Remember you can use the functions defined in your implementation above. #
    ############################################################################
    # Replace "pass" statement with your code
    out_conv, cache_conv = Conv.forward(X, self.params['W1'], self.params['b1'], conv_param)
    out_relu1, cache_relu1 = ReLU.forward(out_conv)
    out_pool, cache_pool = MaxPool.forward(out_relu1, pool_param)

    out_flatten = out_pool.view(out_pool.shape[0], -1)

    out_linear1, cache_linear1 = Linear.forward(out_flatten, self.params['W2'], self.params['b2'])
    out_relu2, cache_relu2 = ReLU.forward(out_linear1)

    scores, cache_linear2 = Linear.forward(out_relu2, self.params['W3'], self.params['b3'])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
        return scores


    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################
    # Replace "pass" statement with your code
    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################
    # Replace "pass" statement with your code
    loss, dscores = softmax_loss(scores, y)

    loss += self.reg * (torch.sum(self.params['W1'] ** 2) +
                        torch.sum(self.params['W2'] ** 2) +
                        torch.sum(self.params['W3'] ** 2))

    dx3, dw3, db3 = Linear.backward(dscores, cache_linear2)
    dx_relu2 = ReLU.backward(dx3, cache_relu2)
    dx2, dw2, db2 = Linear.backward(dx_relu2, cache_linear1)

    dx_pool = dx2.view(out_pool.shape)
    dx_relu1 = MaxPool.backward(dx_pool, cache_pool)
    dx1 = ReLU.backward(dx_relu1, cache_relu1)
    dx, dw1, db1 = Conv.backward(dx1, cache_conv)

    grads['W1'] = dw1 + 2 * self.reg * self.params['W1']
    grads['b1'] = db1
    grads['W2'] = dw2 + 2 * self.reg * self.params['W2']
    grads['b2'] = db2
    grads['W3'] = dw3 + 2 * self.reg * self.params['W3']
    grads['b3'] = db3
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

fix_random_seed(0)
model = ThreeLayerConvNet(**to_double_cuda)

N = 50
X = torch.randn(N, 3, 32, 32, **to_double_cuda)
y = torch.randint(10, size=(N,), **to_long_cuda)

loss, grads = model.loss(X, y)
print('Initial loss (no regularization): ', loss.item())

model.reg = 0.5
loss, grads = model.loss(X, y)
print('Initial loss (with regularization): ', loss.item())

num_inputs = 2
input_dims = (3, 16, 16)
reg = 0.0
num_classes = 10
fix_random_seed(0)
X = torch.randn(num_inputs, *input_dims, **to_double_cuda)
y = torch.randint(num_classes, size=(num_inputs,), **to_long_cuda)

model = ThreeLayerConvNet(num_filters=3, filter_size=3,
                          input_dims=input_dims, hidden_dim=7,
                          weight_scale=5e-2, **to_double_cuda)
loss, grads = model.loss(X, y)

for param_name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    param_grad_num = compute_numeric_gradient(f, model.params[param_name])
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

    fix_random_seed(0)

num_train = 100
small_data = {
  'X_train': data_dict['X_train'][:num_train],
  'y_train': data_dict['y_train'][:num_train],
  'X_val': data_dict['X_val'],
  'y_val': data_dict['y_val'],
}

model = ThreeLayerConvNet(weight_scale=1e-3, **to_float_cuda)

solver = Solver(model, small_data,
                num_epochs=30, batch_size=50,
                update_rule=adam,
                optim_config={
                  'learning_rate': 2e-3,
                },
                verbose=True, print_every=1,
                device='cuda')
solver.train()

plt.title('Training losses')
plt.plot(solver.loss_history, 'o')
plt.xlabel('iteration')
plt.ylabel('loss')
plt.gcf().set_size_inches(9, 4)
plt.show()

plt.title('Train and Val accuracies')
plt.plot(solver.train_acc_history, '-o')
plt.plot(solver.val_acc_history, '-o')
plt.legend(['train', 'val'], loc='upper left')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.gcf().set_size_inches(9, 4)
plt.show()

fix_random_seed(0)

model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001, dtype=torch.float, device='cuda')

solver = Solver(model, data_dict,
                num_epochs=1, batch_size=64,
                update_rule=adam,
                optim_config={
                  'learning_rate': 2e-3,
                },
                verbose=True, print_every=50, device='cuda')
solver.train()

from torchvision.utils import make_grid
nrow = math.ceil(math.sqrt(model.params['W1'].shape[0]))
grid = make_grid(model.params['W1'], nrow=nrow, padding=1, normalize=True, scale_each=True)
plt.imshow(grid.to(device='cpu').permute(1, 2, 0))
plt.axis('off')
plt.gcf().set_size_inches(5, 5)
plt.show()

class DeepConvNet(object):
  """
  A convolutional neural network with an arbitrary number of convolutional
  layers in VGG-Net style. All convolution layers will use kernel size 3 and 
  padding 1 to preserve the feature map size, and all pooling layers will be
  max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
  size of the feature map.

  The network will have the following architecture:
  
  {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

  Each {...} structure is a "macro layer" consisting of a convolution layer,
  an optional batch normalization layer, a ReLU nonlinearity, and an optional
  pooling layer. After L-1 such macro layers, a single fully-connected layer
  is used to predict the class scores.

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
def __init__(self, input_dims=(3, 32, 32),
             num_filters=[8, 8, 8, 8, 8],
             max_pools=[0, 1, 2, 3, 4],
             batchnorm=False,
             num_classes=10, weight_scale=1e-3, reg=0.0,
             weight_initializer=None,
             dtype=torch.float, device='cpu'):
    """
    Initialize a new network.

    Inputs:
    - input_dims: Tuple (C, H, W) giving size of input data
    - num_filters: List of length (L - 1) giving the number of convolutional
      filters to use in each macro layer.
    - max_pools: List of integers giving the indices of the macro layers that
      should have max pooling (zero-indexed).
    - batchnorm: Whether to include batch normalization in each macro layer
    - num_classes: Number of scores to produce from the final linear layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights, or the string "kaiming" to use Kaiming initialization instead
    - reg: Scalar giving L2 regularization strength. L2 regularization should
      only be applied to convolutional and fully-connected weight matrices;
      it should not be applied to biases or to batchnorm scale and shifts.
    - dtype: A torch data type object; all computations will be performed using
      this datatype. float is faster but less accurate, so you should use
      double for numeric gradient checking.
    - device: device to use for computation. 'cpu' or 'cuda'    
    """
    self.params = {}
    self.num_layers = len(num_filters) + 1
    self.max_pools = max_pools
    self.batchnorm = batchnorm
    self.reg = reg
    self.dtype = dtype

    if device == 'cuda':
        device = 'cuda:0'

    ############################################################################
    # TODO: Initialize the parameters for the DeepConvNet. All weights,        #
    # biases, and batchnorm scale and shift parameters should be stored in the #
    # dictionary self.params.                                                  #
    #                                                                          #
    # Weights for conv and fully-connected layers should be initialized        #
    # according to weight_scale. Biases should be initialized to zero.         #
    # Batchnorm scale (gamma) and shift (beta) parameters should be initilized #
    # to ones and zeros respectively.                                          #           
    ############################################################################
    # Replace "pass" statement with your code
    C, H, W = input_dims
    self.params = {}

    prev_channels = C

    for i, num_filter in enumerate(num_filters):
        self.params[f'W{i+1}'] = torch.randn(num_filter, prev_channels, 3, 3, dtype=dtype, device=device) * weight_scale
        self.params[f'b{i+1}'] = torch.zeros(num_filter, dtype=dtype, device=device)

        if batchnorm:
            self.params[f'gamma{i+1}'] = torch.ones(num_filter, dtype=dtype, device=device)
            self.params[f'beta{i+1}'] = torch.zeros(num_filter, dtype=dtype, device=device)

        prev_channels = num_filter

    self.params[f'W{self.num_layers}'] = torch.randn(
        prev_channels * (H // (2 ** len(max_pools))) * (W // (2 ** len(max_pools))),
        num_classes, dtype=dtype, device=device) * weight_scale
    self.params[f'b{self.num_layers}'] = torch.zeros(num_classes, dtype=dtype, device=device)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.batchnorm:
        self.bn_params = [{'mode': 'train'} for _ in range(len(num_filters))]

    # Check that we got the right number of parameters
    if not self.batchnorm:
        params_per_macro_layer = 2  # weight and bias
    else:
        params_per_macro_layer = 4  # weight, bias, scale, shift
    num_params = params_per_macro_layer * len(num_filters) + 2
    msg = 'self.params has the wrong number of elements. Got %d; expected %d'
    msg = msg % (len(self.params), num_params)
    assert len(self.params) == num_params, msg

    # Check that all parameters have the correct device and dtype:
    for k, param in self.params.items():
        msg = 'param "%s" has device %r; should be %r' % (k, param.device, device)
        assert param.device == torch.device(device), msg
        msg = 'param "%s" has dtype %r; should be %r' % (k, param.dtype, dtype)
        assert param.dtype == dtype, msg

def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the deep convolutional network.
    Input / output: Same API as ThreeLayerConvNet.
    """
    X = X.to(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params since they
    # behave differently during training and testing.
    if self.batchnorm:
        for bn_param in self.bn_params:
            bn_param['mode'] = mode
    scores = None

    # pass conv_param to the forward pass for the convolutional layer
    # Padding and stride chosen to preserve the input spatial size
    filter_size = 3
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the DeepConvNet, computing the      #
    # class scores for X and storing them in the scores variable.              #
    #                                                                          #
    # You should use the fast versions of convolution and max pooling layers,  #
    # or the convolutional sandwich layers, to simplify your implementation.   #
    ############################################################################
    # Replace "pass" statement with your code
    out = X
    caches = {}
    pool_caches = {}

    for i in range(1, self.num_layers):
        out, caches[f'conv{i}'] = Conv.forward(out, self.params[f'W{i}'], self.params[f'b{i}'], conv_param)

        if self.batchnorm:
            out, caches[f'bn{i}'] = BatchNorm.forward(out, self.params[f'gamma{i}'], self.params[f'beta{i}'], self.bn_params[i - 1])

        out, caches[f'relu{i}'] = ReLU.forward(out)

        if (i - 1) in self.max_pools:
            out, pool_caches[f'pool{i}'] = MaxPool.forward(out, pool_param)

    out = out.view(out.shape[0], -1)
    scores, caches[f'linear'] = Linear.forward(out, self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}'])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
        return scores


    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the DeepConvNet, storing the loss  #
    # and gradients in the loss and grads variables. Compute data loss using   #
    # softmax, and make sure that grads[k] holds the gradients for             #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization does not include  #
    # a factor of 0.5                                                          #
    ############################################################################
    # Replace "pass" statement with your code
    loss, dscores = softmax_loss(scores, y)

    for i in range(1, self.num_layers + 1):
        loss += self.reg * torch.sum(self.params[f'W{i}'] ** 2)

    dx, dw, db = Linear.backward(dscores, caches[f'linear'])
    grads[f'W{self.num_layers}'] = dw + 2 * self.reg * self.params[f'W{self.num_layers}']
    grads[f'b{self.num_layers}'] = db

    dx = dx.view(-1, *caches[f'relu{self.num_layers - 1}'][0].shape[1:])

    for i in range(self.num_layers - 1, 0, -1):
        if (i - 1) in self.max_pools:
            dx = MaxPool.backward(dx, pool_caches[f'pool{i}'])

        dx = ReLU.backward(dx, caches[f'relu{i}'])

        if self.batchnorm:
            dx, dgamma, dbeta = BatchNorm.backward(dx, caches[f'bn{i}'])
            grads[f'gamma{i}'] = dgamma
            grads[f'beta{i}'] = dbeta

        dx, dw, db = Conv.backward(dx, caches[f'conv{i}'])
        grads[f'W{i}'] = dw + 2 * self.reg * self.params[f'W{i}']
        grads[f'b{i}'] = db
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

fix_random_seed(0)
input_dims = (3, 32, 32)
model = DeepConvNet(num_filters=[8, 64], max_pools=[0, 1], **to_double_cuda)

N = 50
X = torch.randn(N, *input_dims, **to_double_cuda)
y = torch.randint(10, size=(N,), **to_long_cuda)

loss, grads = model.loss(X, y)
print('Initial loss (no regularization): ', loss.item())

model.reg = 1.
loss, grads = model.loss(X, y)
print('Initial loss (with regularization): ', loss.item())

num_inputs = 2
input_dims = (3, 8, 8)
num_classes = 10
fix_random_seed(0)
X = torch.randn(num_inputs, *input_dims, **to_double_cuda)
y = torch.randint(num_classes, size=(num_inputs,), **to_long_cuda)

for reg in [0, 3.14]:
  print('Running check with reg = ', reg)
  model = DeepConvNet(input_dims=input_dims, num_classes=num_classes,
                      num_filters=[8, 8, 8],
                      max_pools=[0, 2],
                      reg=reg,
                      weight_scale=5e-2, **to_double_cuda)

  loss, grads = model.loss(X, y)
  # The relative errors should be up to the order of e-6
  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = compute_numeric_gradient(f, model.params[name])
    print('%s max relative error: %e' % (name, rel_error(grad_num, grads[name])))
  if reg == 0: print()

  # TODO: Use a DeepConvNet to overfit 50 training examples by 
# tweaking just the learning rate and initialization scale.
fix_random_seed(0)

num_train = 50
small_data = {
  'X_train': data_dict['X_train'][:num_train],
  'y_train': data_dict['y_train'][:num_train],
  'X_val': data_dict['X_val'],
  'y_val': data_dict['y_val'],
}
input_dims = small_data['X_train'].shape[1:]

weight_scale = 2e-3   # Experiment with this!
learning_rate = 1e-5  # Experiment with this!
############################################################################
# TODO: Change weight_scale and learning_rate so your model achieves 100%  #
# training accuracy within 30 epochs.                                      #
############################################################################
# Replace "pass" statement with your code
weight_scale = 1e-2   
learning_rate = 5e-4  
############################################################################
#                             END OF YOUR CODE                             #
############################################################################
model = DeepConvNet(input_dims=input_dims, num_classes=10,
                    num_filters=[8, 16, 32, 64],
                    max_pools=[0, 1, 2, 3],
                    reg=1e-5, weight_scale=weight_scale, **to_float_cuda)
solver = Solver(model, small_data,
                print_every=10, num_epochs=30, batch_size=10,
                update_rule=adam,
                optim_config={
                  'learning_rate': learning_rate,
                },
                device='cuda',
         )
solver.train()

plt.plot(solver.loss_history, 'o')
plt.title('Training loss history')
plt.xlabel('Iteration')
plt.ylabel('Training loss')
plt.show()
def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
  """
  Implement Kaiming initialization for linear and convolution layers.
  
  Inputs:
  - Din, Dout: Integers giving the number of input and output dimensions for
    this layer
  - K: If K is None, then initialize weights for a linear layer with Din input
    dimensions and Dout output dimensions. Otherwise if K is a nonnegative
    integer then initialize the weights for a convolution layer with Din input
    channels, Dout output channels, and a kernel size of KxK.
  - relu: If ReLU=True, then initialize weights with a gain of 2 to account for
    a ReLU nonlinearity (Kaiming initializaiton); otherwise initialize weights
    with a gain of 1 (Xavier initialization).
  - device, dtype: The device and datatype for the output tensor.

  Returns:
  - weight: A torch Tensor giving initialized weights for this layer. For a
    linear layer it should have shape (Din, Dout); for a convolution layer it
    should have shape (Dout, Din, K, K).
  """
  gain = 2. if relu else 1.
  weight = None
  if K is None:
    ###########################################################################
    # TODO: Implement Kaiming initialization for linear layer.                #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din).                                   #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    # Replace "pass" statement with your code
    weight_scale = math.sqrt(gain / Din)
    weight = torch.randn(Din, Dout, device=device, dtype=dtype) * weight_scale

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  else:
    ###########################################################################
    # TODO: Implement Kaiming initialization for convolutional layer.         #
    # The weight scale is sqrt(gain / fan_in),                                #
    # where gain is 2 if ReLU is followed by the layer, or 1 if not,          #
    # and fan_in = num_in_channels (= Din) * K * K                            #
    # The output should be a tensor in the designated size, dtype, and device.#
    ###########################################################################
    # Replace "pass" statement with your code
    fan_in = Din * K * K
    weight_scale = math.sqrt(gain / fan_in)
    weight = torch.randn(Dout, Din, K, K, device=device, dtype=dtype) * weight_scale

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  return weight    

fix_random_seed(0)

# Try training a deep convolutional net with different weight initialization methods
num_train = 10000
small_data = {
  'X_train': data_dict['X_train'][:num_train],
  'y_train': data_dict['y_train'][:num_train],
  'X_val': data_dict['X_val'],
  'y_val': data_dict['y_val'],
}
input_dims = data_dict['X_train'].shape[1:]

weight_scales = ['kaiming', 1e-1, 1e-2, 1e-3]

solvers = []
for weight_scale in weight_scales:
  print('Solver with weight scale: ', weight_scale)
  model = DeepConvNet(input_dims=input_dims, num_classes=10,
                      num_filters=([8] * 10) + ([32] * 10) + ([128] * 10),
                      max_pools=[9, 19],
                      weight_scale=weight_scale,
                      reg=1e-5, **to_float_cuda)

  solver = Solver(model, small_data,
                  num_epochs=1, batch_size=128,
                  update_rule=sgd_momentum,
                  optim_config={
                    'learning_rate': 2e-3,
                  },
                  print_every=20, device='cuda')
  solver.train()
  solvers.append(solver)
  
def plot_training_history_init(title, xlabel, solvers, labels, plot_fn, marker='-o'):
    plt.title(title)
    plt.xlabel(xlabel)
    for solver, label in zip(solvers, labels):    
        data = plot_fn(solver)
        label = 'weight_scale=' + str(label)
        plt.plot(data, marker, label=label)
    plt.legend(loc='lower center', ncol=len(solvers))


# Plot training history
plt.subplot(3, 1, 1)
plot_training_history_init('Training loss', 'Iteration', solvers, weight_scales,
                           lambda x: x.loss_history, marker='o')

plt.subplot(3, 1, 2)
plot_training_history_init('Training accuracy', 'Epoch', solvers, weight_scales,
                           lambda x: x.train_acc_history)

plt.subplot(3, 1, 3)
plot_training_history_init('Validation accuracy', 'Epoch', solvers, weight_scales,
                           lambda x: x.val_acc_history)

plt.gcf().set_size_inches(15, 15)
plt.show()

# Pregătire date și rețea
input_dims = data_dict['X_train'].shape[1:]
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

model = None
solver = None

################################################################################
# TODO: Train the best DeepConvNet that you can on CIFAR-10 within 60 seconds. #
################################################################################
# Replace "pass" statement with your code
model = DeepConvNet(
    input_dims=input_dims,
    num_classes=10,
    num_filters=[32, 64, 128, 256],
    max_pools=[0, 1, 2, 3],
    batchnorm=True,
    reg=1e-4,
    weight_scale='kaiming',
    dtype=torch.float32,
    device='cuda'
)

solver = Solver(
    model,
    data_dict,
    num_epochs=10,
    batch_size=128,
    print_every=10,
    update_rule=adam,
    optim_config={'learning_rate': 1e-3},
    device='cuda'
)

################################################################################
#                              END OF YOUR CODE                                #
################################################################################

solver.train(time_limit=60)

torch.backends.cudnn.benchmark = False

print('Validation set accuracy: ', solver.check_accuracy(data_dict['X_val'], data_dict['y_val']))
print('Test set accuracy: ', solver.check_accuracy(data_dict['X_test'], data_dict['y_test']))

class BatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    raise NotImplementedError

  @staticmethod
  def backward(dout, cache):
    raise NotImplementedError

  @staticmethod
  def backward_alt(dout, cache):
    raise NotImplementedError

def plot_training_history_init(title, xlabel, solvers, labels, plot_fn, marker='-o'):
    plt.title(title)
    plt.xlabel(xlabel)
    for solver, label in zip(solvers, labels):    
        data = plot_fn(solver)
        label = 'weight_scale=' + str(label)
        plt.plot(data, marker, label=label)
    plt.legend(loc='lower center', ncol=len(solvers))


# Plot training history
plt.subplot(3, 1, 1)
plot_training_history_init('Training loss', 'Iteration', solvers, weight_scales,
                           lambda x: x.loss_history, marker='o')

plt.subplot(3, 1, 2)
plot_training_history_init('Training accuracy', 'Epoch', solvers, weight_scales,
                           lambda x: x.train_acc_history)

plt.subplot(3, 1, 3)
plot_training_history_init('Validation accuracy', 'Epoch', solvers, weight_scales,
                           lambda x: x.val_acc_history)

plt.gcf().set_size_inches(15, 15)
plt.show()

# Pregătire date și rețea
input_dims = data_dict['X_train'].shape[1:]
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

model = None
solver = None

################################################################################
# TODO: Train the best DeepConvNet that you can on CIFAR-10 within 60 seconds. #
################################################################################
# Replace "pass" statement with your code
model = DeepConvNet(
    input_dims=input_dims,
    num_classes=10,
    num_filters=[32, 64, 128, 256],
    max_pools=[0, 1, 2, 3],
    batchnorm=True,
    reg=1e-4,
    weight_scale='kaiming',
    dtype=torch.float32,
    device='cuda'
)

solver = Solver(
    model,
    data_dict,
    num_epochs=10,
    batch_size=128,
    print_every=10,
    update_rule=adam,
    optim_config={'learning_rate': 1e-3},
    device='cuda'
)

################################################################################
#                              END OF YOUR CODE                                #
################################################################################

x_norm = (x - running_mean) / torch.sqrt(running_var + eps)
out = gamma * x_norm + beta
cache = None

#######################################################################
#                           END OF YOUR CODE                          #
#######################################################################

BatchNorm.forward = batchnorm_forward

# Check the training-time forward pass by checking means and variances
# of features both before and after batch normalization   
def print_mean_std(x,dim=0):
  means = ['%.3f' % xx for xx in x.mean(dim=dim).tolist()]
  stds = ['%.3f' % xx for xx in x.std(dim=dim).tolist()]
  print('  means: ', means)
  print('  stds:  ', stds)
  print()

# Simulate the forward pass for a two-layer network
fix_random_seed(0)
N, D1, D2, D3 = 200, 50, 60, 3
X = torch.randn(N, D1, **to_double_cuda)
W1 = torch.randn(D1, D2, **to_double_cuda)
W2 = torch.randn(D2, D3, **to_double_cuda)
a = X.matmul(W1).clamp(min=0.).matmul(W2)

print('Before batch normalization:')
print_mean_std(a,dim=0)

# Run with gamma=1, beta=0. Means should be close to zero and stds close to one
gamma = torch.ones(D3, **to_double_cuda)
beta = torch.zeros(D3, **to_double_cuda)
print('After batch normalization (gamma=1, beta=0)')
a_norm, _ = BatchNorm.forward(a, gamma, beta, {'mode': 'train'})
print_mean_std(a_norm,dim=0)

# Run again with nontrivial gamma and beta. Now means should be close to beta
# and std should be close to gamma.
gamma = torch.tensor([1.0, 2.0, 3.0], **to_double_cuda)
beta = torch.tensor([11.0, 12.0, 13.0], **to_double_cuda)
print('After batch normalization (gamma=', gamma.tolist(), ', beta=', beta.tolist(), ')')
a_norm, _ = BatchNorm.forward(a, gamma, beta, {'mode': 'train'})
print_mean_std(a_norm,dim=0)

fix_random_seed(0)
N, D1, D2, D3 = 200, 50, 60, 3
W1 = torch.randn(D1, D2, **to_double_cuda)
W2 = torch.randn(D2, D3, **to_double_cuda)

bn_param = {'mode': 'train'}
gamma = torch.ones(D3, **to_double_cuda)
beta = torch.zeros(D3, **to_double_cuda)

for t in range(500):
  X = torch.randn(N, D1, **to_double_cuda)
  a = X.matmul(W1).clamp(min=0.).matmul(W2)
  BatchNorm.forward(a, gamma, beta, bn_param)

bn_param['mode'] = 'test'
X = torch.randn(N, D1, **to_double_cuda)
a = X.matmul(W1).clamp(min=0.).matmul(W2)
a_norm, _ = BatchNorm.forward(a, gamma, beta, bn_param)

# Means should be close to zero and stds close to one, but will be
# noisier than training-time forward passes.
print('After batch normalization (test-time):')
print_mean_std(a_norm,dim=0)

def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    # Don't forget to implement train and test mode separately.               #
    ###########################################################################
    # Replace "pass" statement with your code
    x, x_norm, gamma, beta, sample_mean, sample_var, eps = cache
    N, D = dout.shape

    dbeta = dout.sum(dim=0)
    dgamma = (dout * x_norm).sum(dim=0)

    dx_norm = dout * gamma
    dvar = (dx_norm * (x - sample_mean) * -0.5 * torch.pow(sample_var + eps, -1.5)).sum(dim=0)
    dmean = (dx_norm * -1 / torch.sqrt(sample_var + eps)).sum(dim=0) + dvar * (-2 * (x - sample_mean)).sum(dim=0) / N

    dx = dx_norm / torch.sqrt(sample_var + eps) + dvar * 2 * (x - sample_mean) / N + dmean / N
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


BatchNorm.backward = batchnorm_backward

# Gradient check batchnorm backward pass
fix_random_seed(0)
N, D = 4, 5
x = 5 * torch.randn(N, D, **to_double_cuda) + 12
gamma = torch.randn(D, **to_double_cuda)
beta = torch.randn(D, **to_double_cuda)
dout = torch.randn(N, D, **to_double_cuda)

bn_param = {'mode': 'train'}
fx = lambda x: BatchNorm.forward(x, gamma, beta, bn_param)[0]
fg = lambda a: BatchNorm.forward(x, a, beta, bn_param)[0]
fb = lambda b: BatchNorm.forward(x, gamma, b, bn_param)[0]

dx_num = compute_numeric_gradient(fx, x, dout)
da_num = compute_numeric_gradient(fg, gamma.clone(), dout)
db_num = compute_numeric_gradient(fb, beta.clone(), dout)

_, cache = BatchNorm.forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = BatchNorm.backward(dout, cache)
# You should expect to see relative errors between 1e-12 and 1e-9
print('dx error: ', rel_error(dx_num, dx))
print('dgamma error: ', rel_error(da_num, dgamma))
print('dbeta error: ', rel_error(db_num, dbeta))

def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.
    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass. 
    See the jupyter notebook for more hints.
    
    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # Replace "pass" statement with your code
    x, x_norm, gamma, beta, sample_mean, sample_var, eps = cache
    N, D = dout.shape

    dbeta = dout.sum(dim=0)
    dgamma = (dout * x_norm).sum(dim=0)

    dx = (gamma / (N * torch.sqrt(sample_var + eps))) * (
        N * dout - dbeta - x_norm * dgamma
    )
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta

BatchNorm.backward_alt = batchnorm_backward_alt

fix_random_seed(0)
N, D = 128, 2048
x = 5 * torch.randn(N, D, **to_double_cuda) + 12
gamma = torch.randn(D, **to_double_cuda)
beta = torch.randn(D, **to_double_cuda)
dout = torch.randn(N, D, **to_double_cuda)

bn_param = {'mode': 'train'}
out, cache = BatchNorm.forward(x, gamma, beta, bn_param)

t1 = time.time()
dx1, dgamma1, dbeta1 = BatchNorm.backward(dout, cache)
t2 = time.time()
dx2, dgamma2, dbeta2 = BatchNorm.backward_alt(dout, cache)
t3 = time.time()

print('dx difference: ', rel_error(dx1, dx2))
print('dgamma difference: ', rel_error(dgamma1, dgamma2))
print('dbeta difference: ', rel_error(dbeta1, dbeta2))
print('speedup: %.2fx' % ((t2 - t1) / (t3 - t2)))

class SpatialBatchNorm(object):

  @staticmethod
  def forward(x, gamma, beta, bn_param):
    raise NotImplementedError

  @staticmethod
  def backward(dout, cache):
    raise NotImplementedError

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
    old information is discarded completely at every time step, while
    momentum=1 means that new information is never incorporated. The
    default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  ###########################################################################
  # TODO: Implement the forward pass for spatial batch normalization.       #
  #                                                                         #
  # HINT: You can implement spatial batch normalization by calling the      #
  # vanilla version of batch normalization you implemented above.           #
  # Your implementation should be very short; ours is less than five lines. #
  ###########################################################################
  # Replace "pass" statement with your code
  N, C, H, W = x.shape
  x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)
  out_reshaped, cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
  out = out_reshaped.reshape(N, H, W, C).permute(0, 3, 1, 2)

  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################

  return out, cache

SpatialBatchNorm.forward = spatial_batchnorm_forward

fix_random_seed(0)
# Check the training-time forward pass by checking means and variances
# of features both before and after spatial batch normalization

N, C, H, W = 2, 3, 4, 5
x = 4 * torch.randn(N, C, H, W, **to_double_cuda) + 10

print('Before spatial batch normalization:')
print('  Shape: ', x.shape)
print('  Means: ', x.mean(dim=(0, 2, 3)))
print('  Stds: ', x.std(dim=(0, 2, 3)))

# Means should be close to zero and stds close to one
gamma, beta = torch.ones(C, **to_double_cuda), torch.zeros(C, **to_double_cuda)
bn_param = {'mode': 'train'}
out, _ = SpatialBatchNorm.forward(x, gamma, beta, bn_param)
print('After spatial batch normalization:')
print('  Shape: ', out.shape)
print('  Means: ', out.mean(dim=(0, 2, 3)))
print('  Stds: ', out.std(dim=(0, 2, 3)))

# Means should be close to beta and stds close to gamma
gamma, beta = torch.tensor([3, 4, 5], **to_double_cuda), torch.tensor([6, 7, 8], **to_double_cuda)
out, _ = SpatialBatchNorm.forward(x, gamma, beta, bn_param)
print('After spatial batch normalization (nontrivial gamma, beta):')
print('  Shape: ', out.shape)
print('  Means: ', out.mean(dim=(0, 2, 3)))
print('  Stds: ', out.std(dim=(0, 2, 3)))

fix_random_seed(0)
# Check the test-time forward pass by running the training-time
# forward pass many times to warm up the running averages, and then
# checking the means and variances of activations after a test-time
# forward pass.
N, C, H, W = 10, 4, 11, 12

bn_param = {'mode': 'train'}
gamma = torch.ones(C, **to_double_cuda)
beta = torch.zeros(C, **to_double_cuda)
for t in range(50):
  x = 2.3 * torch.randn(N, C, H, W, **to_double_cuda) + 13
  SpatialBatchNorm.forward(x, gamma, beta, bn_param)
bn_param['mode'] = 'test'
x = 2.3 * torch.randn(N, C, H, W, **to_double_cuda) + 13
a_norm, _ = SpatialBatchNorm.forward(x, gamma, beta, bn_param)

# Means should be close to zero and stds close to one, but will be
# noisier than training-time forward passes.
print('After spatial batch normalization (test-time):')
print('  means: ', a_norm.mean(dim=(0, 2, 3)))
print('  stds: ', a_norm.std(dim=(0, 2, 3)))

def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  ###########################################################################
  # TODO: Implement the backward pass for spatial batch normalization.      #
  #                                                                         #
  # HINT: You can implement spatial batch normalization by calling the      #
  # vanilla version of batch normalization you implemented above.           #
  # Your implementation should be very short; ours is less than five lines. #
  ###########################################################################
  # Replace "pass" statement with your code
  N, C, H, W = dout.shape
  dout_reshaped = dout.permute(0, 2, 3, 1).reshape(-1, C)
  dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)
  dx = dx_reshaped.reshape(N, H, W, C).permute(0, 3, 1, 2)

  ###########################################################################
  #                             END OF YOUR CODE                            #
  ###########################################################################

  return dx, dgamma, dbeta

SpatialBatchNorm.backward = spatial_batchnorm_backward

fix_random_seed(0)
N, C, H, W = 2, 3, 4, 5
x = 5 * torch.randn(N, C, H, W, **to_double_cuda) + 12
gamma = torch.randn(C, **to_double_cuda)
beta = torch.randn(C, **to_double_cuda)
dout = torch.randn(N, C, H, W, **to_double_cuda)

bn_param = {'mode': 'train'}
fx = lambda x: SpatialBatchNorm.forward(x, gamma, beta, bn_param)[0]
fg = lambda a: SpatialBatchNorm.forward(x, gamma, beta, bn_param)[0]
fb = lambda b: SpatialBatchNorm.forward(x, gamma, beta, bn_param)[0]

dx_num = compute_numeric_gradient(fx, x, dout)
da_num = compute_numeric_gradient(fg, gamma, dout)
db_num = compute_numeric_gradient(fb, beta, dout)

_, cache = SpatialBatchNorm.forward(x, gamma, beta, bn_param)
dx, dgamma, dbeta = SpatialBatchNorm.backward(dout, cache)
print('dx error: ', rel_error(dx_num, dx))
print('dgamma error: ', rel_error(da_num, dgamma))
print('dbeta error: ', rel_error(db_num, dbeta))

class Linear_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, bn_param):
    """
    Convenience layer that performs an linear transform, batch normalization,
    and ReLU.
    Inputs:
    - x: Array of shape (N, D1); input to the linear layer
    - w, b: Arrays of shape (D2, D2) and (D2,) giving the weight and bias for
      the linear transform.
    - gamma, beta: Arrays of shape (D2,) and (D2,) giving scale and shift
      parameters for batch normalization.
    - bn_param: Dictionary of parameters for batch normalization.
    Returns:
    - out: Output from ReLU, of shape (N, D2)
    - cache: Object to give to the backward pass.
    """
    a, fc_cache = Linear.forward(x, w, b)
    a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    """
    Backward pass for the linear-batchnorm-relu convenience layer.
    """
    fc_cache, bn_cache, relu_cache = cache
    da_bn = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
    dx, dw, db = Linear.backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    out, relu_cache = ReLU.forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = ReLU.backward(dout, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

  @staticmethod
  def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
    a, conv_cache = FastConv.forward(x, w, b, conv_param)
    an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
    s, relu_cache = ReLU.forward(an)
    out, pool_cache = FastMaxPool.forward(s, pool_param)
    cache = (conv_cache, bn_cache, relu_cache, pool_cache)
    return out, cache

  @staticmethod
  def backward(dout, cache):
    conv_cache, bn_cache, relu_cache, pool_cache = cache
    ds = FastMaxPool.backward(dout, pool_cache)
    dan = ReLU.backward(ds, relu_cache)
    da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
    dx, dw, db = FastConv.backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta

num_inputs = 2
input_dims = (3, 8, 8)
num_classes = 10
fix_random_seed(0)
X = torch.randn(num_inputs, *input_dims, **to_double_cuda)
y = torch.randint(num_classes, size=(num_inputs,), **to_long_cuda)

for reg in [0, 3.14]:
  print('Running check with reg = ', reg)
  model = DeepConvNet(input_dims=input_dims, num_classes=num_classes,
                      num_filters=[8, 8, 8],
                      max_pools=[0, 2],
                      reg=reg, batchnorm=True,
                      weight_scale='kaiming',
                      **to_double_cuda)

  loss, grads = model.loss(X, y)
  # The relative errors should be up to the order of e-3
  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = compute_numeric_gradient(f, model.params[name])
    print('%s max relative error: %e' % (name, rel_error(grad_num, grads[name])))
  if reg == 0: print()
  
  fix_random_seed(0)

# Try training a deep convolutional net with batchnorm
num_train = 500
small_data = {
  'X_train': data_dict['X_train'][:num_train],
  'y_train': data_dict['y_train'][:num_train],
  'X_val': data_dict['X_val'],
  'y_val': data_dict['y_val'],
}
input_dims = data_dict['X_train'].shape[1:]

bn_model = DeepConvNet(input_dims=input_dims, num_classes=10,
                       num_filters=[16, 32, 32, 64, 64],
                       max_pools=[0, 1, 2, 3, 4],
                       weight_scale='kaiming',
                       batchnorm=True,
                       reg=1e-5, **to_float_cuda)
model = DeepConvNet(input_dims=input_dims, num_classes=10,
                    num_filters=[16, 32, 32, 64, 64],
                    max_pools=[0, 1, 2, 3, 4],
                    weight_scale='kaiming',
                    batchnorm=False,
                    reg=1e-5, **to_float_cuda)

print('Solver with batch norm:')
bn_solver = Solver(bn_model, small_data,
                   num_epochs=10, batch_size=100,
                   update_rule=adam,
                   optim_config={
                     'learning_rate': 1e-3,
                   },
                   print_every=20, device='cuda')
bn_solver.train()

print('\nSolver without batch norm:')
solver = Solver(model, small_data,
                num_epochs=10, batch_size=100,
                update_rule=adam,
                optim_config={
                  'learning_rate': 1e-3,
                },
                print_every=20, device='cuda')
solver.train()

def plot_training_history_bn(title, label, solvers, bn_solvers, plot_fn, bl_marker='.', bn_marker='.', labels=None):
  """utility function for plotting training history"""
  plt.title(title)
  plt.xlabel(label)
  bn_plots = [plot_fn(bn_solver) for bn_solver in bn_solvers]
  bl_plots = [plot_fn(solver) for solver in solvers]
  num_bn = len(bn_plots)
  num_bl = len(bl_plots)
  for i in range(num_bn):
    label='w/ BN'
    if labels is not None:
      label += str(labels[i])
    plt.plot(bn_plots[i], bn_marker, label=label)
  for i in range(num_bl):
    label='w/o BN'
    if labels is not None:
      label += str(labels[i])
    plt.plot(bl_plots[i], bl_marker, label=label)
  plt.legend(loc='lower center', ncol=num_bn+num_bl)
  
  plt.subplot(3, 1, 1)
plot_training_history_bn('Training loss','Iteration', [solver], [bn_solver], \
                      lambda x: x.loss_history, bl_marker='-o', bn_marker='-o')
plt.subplot(3, 1, 2)
plot_training_history_bn('Training accuracy','Epoch', [solver], [bn_solver], \
                      lambda x: x.train_acc_history, bl_marker='-o', bn_marker='-o')
plt.subplot(3, 1, 3)
plot_training_history_bn('Validation accuracy','Epoch', [solver], [bn_solver], \
                      lambda x: x.val_acc_history, bl_marker='-o', bn_marker='-o')

plt.gcf().set_size_inches(15, 15)
plt.show()

fix_random_seed(0)

# Try training a very deep net with batchnorm
num_train = 10000
small_data = {
  'X_train': data_dict['X_train'][:num_train],
  'y_train': data_dict['y_train'][:num_train],
  'X_val': data_dict['X_val'],
  'y_val': data_dict['y_val'],
}
input_dims = data_dict['X_train'].shape[1:]
num_epochs = 5
lrs = [2e-1, 1e-1, 5e-2]
lrs = [5e-3, 1e-2, 2e-2]

solvers = []
for lr in lrs:
  print('No normalization: learning rate = ', lr)
  model = DeepConvNet(input_dims=input_dims, num_classes=10,
                      num_filters=[8, 8, 8],
                      max_pools=[0, 1, 2],
                      weight_scale='kaiming',
                      batchnorm=False,
                      reg=1e-5, **to_float_cuda)
  solver = Solver(model, small_data,
                  num_epochs=num_epochs, batch_size=100,
                  update_rule=sgd_momentum,
                  optim_config={
                    'learning_rate': lr,
                  },
                  verbose=False, device='cuda')
  solver.train()
  solvers.append(solver)

bn_solvers = []
for lr in lrs:
  print('Normalization: learning rate = ', lr)
  bn_model = DeepConvNet(input_dims=input_dims, num_classes=10,
                         num_filters=[8, 8, 16, 16, 32, 32],
                         max_pools=[1, 3, 5],
                         weight_scale='kaiming',
                         batchnorm=True,
                         reg=1e-5, **to_float_cuda)
  bn_solver = Solver(bn_model, small_data,
                     num_epochs=num_epochs, batch_size=128,
                     update_rule=sgd_momentum,
                     optim_config={
                       'learning_rate': lr,
                     },
                     verbose=False, device='cuda')
  bn_solver.train()
  bn_solvers.append(bn_solver)
  
  plt.subplot(2, 1, 1)
plot_training_history_bn('Training accuracy (Batch Normalization)','Epoch', solvers, bn_solvers, \
                      lambda x: x.train_acc_history, bl_marker='-^', bn_marker='-o', labels=[' lr={:.0e}'.format(lr) for lr in lrs])
plt.subplot(2, 1, 2)
plot_training_history_bn('Validation accuracy (Batch Normalization)','Epoch', solvers, bn_solvers, \
                      lambda x: x.val_acc_history, bl_marker='-^', bn_marker='-o', labels=[' lr={:.0e}'.format(lr) for lr in lrs])

plt.gcf().set_size_inches(15, 10)
plt.show()