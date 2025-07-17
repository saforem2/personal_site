# Intro to NNs: MNIST
Sam Foreman, Marieme Ngom, Huihuo Zheng, Bethany Lusch, Taylor Childers
2025-07-17

<link rel="preconnect" href="https://fonts.googleapis.com">

- [The MNIST dataset](#the-mnist-dataset)
- [Generalities:](#generalities)
- [Linear Model](#linear-model)
- [Learning](#learning)
- [Prediction](#prediction)
- [Multilayer Model](#multilayer-model)
- [Important things to know](#important-things-to-know)
- [Recap](#recap)
- [Homework](#homework)
- [Homework solution](#homework-solution)

Author: Marieme Ngom, adapting materials from Bethany Lusch, Asad Khan,
Prasanna Balaprakash, Taylor Childers, Corey Adams, Kyle Felker, and
Tanwi Mallick.

This tutorial will serve as a gentle introduction to neural networks and
deep learning through a hands-on classification problem using the MNIST
dataset.

In particular, we will introduce neural networks and how to train and
improve their learning capabilities. We will use the PyTorch Python
library.

The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) contains
thousands of examples of handwritten numbers, with each digit labeled
0-9.

<div id="fig-mnist-task">

<img src="../images/mnist_task.png" width="400" />

Figure 1: MNIST sample

</div>

``` python
%matplotlib inline

import torch
import torchvision
from torch import nn

import numpy 
import matplotlib.pyplot as plt
import time
```

## The MNIST dataset

We will now download the dataset that contains handwritten digits. MNIST
is a popular dataset, so we can download it via the PyTorch library.
Note: - x is for the inputs (images of handwritten digits) and y is for
the labels or outputs (digits 0-9) - We are given “training” and “test”
datasets. Training datasets are used to fit the model. Test datasets are
saved until the end, when we are satisfied with our model, to estimate
how well our model generalizes to new data.

Note that downloading it the first time might take some time. The data
is split as follows: - 60,000 training examples, 10,000 test examples -
inputs: 1 x 28 x 28 pixels - outputs (labels): one integer per example

``` python
training_data = torchvision.datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

test_data = torchvision.datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)
```

``` python
train_size = int(0.8 * len(training_data))  # 80% for training
val_size = len(training_data) - train_size  # Remaining 20% for validation
training_data, validation_data = torch.utils.data.random_split(
    training_data,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(55)
)
```

``` python
print('MNIST data loaded: train:',len(training_data),' examples, validation: ', len(validation_data), 'examples, test:',len(test_data), 'examples')
print('Input shape', training_data[0][0].shape)
```

    MNIST data loaded: train: 48000  examples, validation:  12000 examples, test: 10000 examples
    Input shape torch.Size([1, 28, 28])

Let’s take a closer look. Here are the first 10 training digits:

``` python
pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(numpy.reshape(training_data[i][0], (28, 28)), cmap="gray")
    plt.title('Class: '+str(training_data[i][1]))
```

    Text(0.5, 1.0, 'Class: 1')

    Text(0.5, 1.0, 'Class: 0')

    Text(0.5, 1.0, 'Class: 2')

    Text(0.5, 1.0, 'Class: 0')

    Text(0.5, 1.0, 'Class: 9')

    Text(0.5, 1.0, 'Class: 7')

    Text(0.5, 1.0, 'Class: 7')

    Text(0.5, 1.0, 'Class: 1')

    Text(0.5, 1.0, 'Class: 0')

    Text(0.5, 1.0, 'Class: 8')

![](index_files/figure-commonmark/cell-6-output-11.png)

## Generalities:

To train our classifier, we need (besides the data): - A model that
depend on parameters $\mathbf{\theta}$. Here we are going to use neural
networks. - A loss function $J(\mathbf{\theta})$ to measure the
capabilities of the model. - An optimization method.

## Linear Model

Let’s begin with a simple linear model: linear regression, like last
week. We add one complication: each example is a vector (flattened
image), so the “slope” multiplication becomes a dot product. If the
target output is a vector as well, then the multiplication becomes
matrix multiplication.

Note, like before, we consider multiple examples at once, adding another
dimension to the input.

<div id="fig-linear-model">

![](../images/LinearModel_1.png)

Figure 2: Linear model for classification

</div>

The linear layers in PyTorch perform a basic $xW + b$. These “fully
connected” layers connect each input to each output with some weight
parameter. We wouldn’t expect a simple linear model $f(x) = xW+b$
directly outputting the class label and minimizing mean squared error to
work well - the model would output labels like 3.55 and 2.11 instead of
skipping to integers.

We now need: - A loss function $J(\theta)$ where $\theta$ is the list of
parameters (here W and b). Last week, we used mean squared error (MSE),
but this week let’s make two changes that make more sense for
classification: - Change the output to be a length-10 vector of class
probabilities (0 to 1, adding to 1). - Cross entropy as the loss
function, which is typical for classification. You can read more
[here](https://gombru.github.io/2018/05/23/cross_entropy_loss/).

- An optimization method or optimizer such as the stochastic gradient
  descent (sgd) method, the Adam optimizer, RMSprop, Adagrad etc. Let’s
  start with stochastic gradient descent (sgd), like last week. For far
  more information about more advanced optimizers than basic SGD, with
  some cool animations, see
  https://ruder.io/optimizing-gradient-descent/ or
  https://distill.pub/2017/momentum/.

- A learning rate. As we learned last week, the learning rate controls
  how far we move during each step.

``` python
class LinearClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        
        # First, we need to convert the input image to a vector by using 
        # nn.Flatten(). For MNIST, it means the second dimension 28*28 becomes 784.
        self.flatten = nn.Flatten()
        
        # Here, we add a fully connected ("dense") layer that has 28 x 28 = 784 input nodes 
        #(one for each pixel in the input image) and 10 output nodes (for probabilities of each class).
        self.layer_1 = nn.Linear(28*28, 10)
        
    def forward(self, x):

        x = self.flatten(x)
        x = self.layer_1(x)

        return x
```

``` python
linear_model = LinearClassifier()
print(linear_model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.05)
```

    LinearClassifier(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (layer_1): Linear(in_features=784, out_features=10, bias=True)
    )

## Learning

Now we are ready to train our first model. A training step is comprised
of: - A forward pass: the input is passed through the network -
Backpropagation: A backward pass to compute the gradient
$\frac{\partial J}{\partial \mathbf{W}}$ of the loss function with
respect to the parameters of the network. - Weight updates \$ =  - \$
where $\alpha$ is the learning rate.

How many steps do we take? - The batch size corresponds to the number of
training examples in one pass (forward + backward). A smaller batch size
allows the model to learn from individual examples but takes longer to
train. A larger batch size requires fewer steps but may result in the
model not capturing the nuances in the data. The higher the batch size,
the more memory you will require.  
- An epoch means one pass through the whole training data (looping over
the batches). Using few epochs can lead to underfitting and using too
many can lead to overfitting. - The choice of batch size and learning
rate are important for performance, generalization and accuracy in deep
learning.

``` python
batch_size = 128

# The dataloader makes our dataset iterable 
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
val_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
```

``` python
def train_one_epoch(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # backward pass calculates gradients
        loss.backward()
        
        # take one step with these gradients
        optimizer.step()
        
        # resets the gradients 
        optimizer.zero_grad()
```

``` python
def evaluate(dataloader, model, loss_fn):
    # Set the model to evaluation mode - some NN pieces behave differently during training
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    loss, correct = 0, 0

    # We can save computation and memory by not calculating gradients here - we aren't optimizing 
    with torch.no_grad():
        # loop over all of the batches
        for X, y in dataloader:
            pred = model(X)
            loss += loss_fn(pred, y).item()
            # how many are correct in this batch? Tracking for accuracy 
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    loss /= num_batches
    correct /= size
    
    accuracy = 100*correct
    return accuracy, loss
```

``` python
%%time

epochs = 5
train_acc_all = []
val_acc_all = []
for j in range(epochs):
    train_one_epoch(train_dataloader, linear_model, loss_fn, optimizer)
    
    # checking on the training loss and accuracy once per epoch
    acc, loss = evaluate(train_dataloader, linear_model, loss_fn)
    train_acc_all.append(acc)
    print(f"Epoch {j}: training loss: {loss}, accuracy: {acc}")
    
    # checking on the validation loss and accuracy once per epoch
    val_acc, val_loss = evaluate(val_dataloader, linear_model, loss_fn)
    val_acc_all.append(val_acc)
    print(f"Epoch {j}: val. loss: {val_loss}, val. accuracy: {val_acc}")
```

    Epoch 0: training loss: 0.5022523045539856, accuracy: 87.53958333333334
    Epoch 0: val. loss: 0.4948802136994423, val. accuracy: 87.61666666666666
    Epoch 1: training loss: 0.4217463522354762, accuracy: 88.99583333333332
    Epoch 1: val. loss: 0.4127243398985964, val. accuracy: 88.85
    Epoch 2: training loss: 0.3877290372848511, accuracy: 89.66875
    Epoch 2: val. loss: 0.3780994909874936, val. accuracy: 89.47500000000001
    Epoch 3: training loss: 0.36773853107293447, accuracy: 90.07083333333334
    Epoch 3: val. loss: 0.357905276119709, val. accuracy: 89.90833333333333
    Epoch 4: training loss: 0.35414402723312377, accuracy: 90.39166666666667
    Epoch 4: val. loss: 0.34430239904434123, val. accuracy: 90.24166666666666
    CPU times: user 26.1 s, sys: 20.6 s, total: 46.7 s
    Wall time: 12.9 s

``` python
pltsize=1
plt.figure(figsize=(10*pltsize, 10 * pltsize))
plt.plot(range(epochs), train_acc_all,label = 'Training Acc.' )
plt.plot(range(epochs), val_acc_all, label = 'Validation Acc.' )
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
```

    Text(0.5, 0, 'Epoch #')

    Text(0, 0.5, 'Loss')

![](index_files/figure-commonmark/cell-13-output-3.png)

``` python
# Visualize how the model is doing on the first 10 examples
pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))
linear_model.eval()
batch = next(iter(train_dataloader))
predictions = linear_model(batch[0])

for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    plt.imshow(batch[0][i,0,:,:], cmap="gray")
    plt.title('%d' % predictions[i,:].argmax())
```

    LinearClassifier(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (layer_1): Linear(in_features=784, out_features=10, bias=True)
    )

    Text(0.5, 1.0, '1')

    Text(0.5, 1.0, '0')

    Text(0.5, 1.0, '2')

    Text(0.5, 1.0, '0')

    Text(0.5, 1.0, '9')

    Text(0.5, 1.0, '7')

    Text(0.5, 1.0, '7')

    Text(0.5, 1.0, '1')

    Text(0.5, 1.0, '0')

    Text(0.5, 1.0, '8')

![](index_files/figure-commonmark/cell-14-output-12.png)

Exercise: How can you improve the accuracy? Some things you might
consider: increasing the number of epochs, changing the learning rate,
etc.

## Prediction

Let’s see how our model generalizes to the unseen test data.

``` python
#For HW: cell to change batch size
#create dataloader for test data
# The dataloader makes our dataset iterable

batch_size_test = 256 
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test)
```

``` python
acc_test, loss_test = evaluate(test_dataloader, linear_model, loss_fn)
print("Test loss: %.4f, test accuracy: %.2f%%" % (loss_test, acc_test))
```

    Test loss: 0.3323, test accuracy: 90.89%

We can now take a closer look at the results.

Let’s define a helper function to show the failure cases of our
classifier.

``` python
def show_failures(model, dataloader, maxtoshow=10):
    model.eval()
    batch = next(iter(dataloader))
    predictions = model(batch[0])
    
    rounded = predictions.argmax(1)
    errors = rounded!=batch[1]
    print('Showing max', maxtoshow, 'first failures. '
          'The predicted class is shown first and the correct class in parentheses.')
    ii = 0
    plt.figure(figsize=(maxtoshow, 1))
    for i in range(batch[0].shape[0]):
        if ii>=maxtoshow:
            break
        if errors[i]:
            plt.subplot(1, maxtoshow, ii+1)
            plt.axis('off')
            plt.imshow(batch[0][i,0,:,:], cmap="gray")
            plt.title("%d (%d)" % (rounded[i], batch[1][i]))
            ii = ii + 1
```

Here are the first 10 images from the test data that this small model
classified to a wrong class:

``` python
show_failures(linear_model, test_dataloader)
```

    Showing max 10 first failures. The predicted class is shown first and the correct class in parentheses.

![](index_files/figure-commonmark/cell-18-output-2.png)

## Multilayer Model

Our linear model isn’t enough for high accuracy on this dataset. To
improve the model, we often need to add more layers and nonlinearities.

<div id="fig-shallow-nn">

![](../images/shallow_nn.png)

Figure 3: Shallow neural network

</div>

The output of this NN can be written as where $\mathbf{x}$ is the input,
$\mathbf{W}_j$ are the weights of the neural network, $\sigma_j$ the
(nonlinear) activation functions, and $\mathbf{b}_j$ its biases. The
activation function introduces the nonlinearity and makes it possible to
learn more complex tasks. Desirable properties in an activation function
include being differentiable, bounded, and monotonic.

Image source:
[PragatiBaheti](https://www.v7labs.com/blog/neural-networks-activation-functions)

<div id="fig-activation">

![](../images/activation.jpeg)

Figure 4: Activation function

</div>

Adding more layers to obtain a deep neural network:

<div id="fig-nn-annotated">

![](../images/deep_nn_annotated.jpg)

Figure 5

</div>

## Important things to know

Deep Neural networks can be overly flexible/complicated and “overfit”
your data, just like fitting overly complicated polynomials:

<div id="fig-bias-variance">

![](../images/bias_vs_variance.png)

Figure 6: Bias-variance tradeoff

</div>

Vizualization wrt to the accuracy and loss (Image source:
[Baeldung](https://www.baeldung.com/cs/ml-underfitting-overfitting)):

<div id="fig-acc-under-over">

![](./images/acc_under_over.webp)

Figure 7: Visualization of accuracy and loss

</div>

To improve the generalization of our model on previously unseen data, we
employ a technique known as regularization, which constrains our
optimization problem in order to discourage complex models.

- Dropout is the commonly used regularization technique. The Dropout
  layer randomly sets input units to 0 with a frequency of rate at each
  step during training time, which helps prevent overfitting.
- Penalizing the loss function by adding a term such as
  $\lambda ||\mathbf{W}||^2$ is alsp a commonly used regularization
  technique. This helps “control” the magnitude of the weights of the
  network.

Vanishing gradients  
Gradients become small as they propagate backward through the layers.

Squashing activation functions like sigmoid or tanh could cause this.

Exploding gradients  
Gradients grow exponentially usually due to “poor” weight
initialization.

We can now implement a deep network in PyTorch.

`nn.Dropout()` performs the Dropout operation mentioned earlier:

``` python
#For HW: cell to change activation
class NonlinearClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers_stack = nn.Sequential(
            nn.Linear(28*28, 50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 50),
            nn.ReLU(),
           # nn.Dropout(0.2),
            nn.Linear(50, 50),
            nn.ReLU(),
           # nn.Dropout(0.2),
            nn.Linear(50, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers_stack(x)

        return x
```

``` python
#### For HW: cell to change learning rate
nonlinear_model = NonlinearClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(nonlinear_model.parameters(), lr=0.05)
```

``` python
%%time

epochs = 5
train_acc_all = []
val_acc_all = []
for j in range(epochs):
    train_one_epoch(train_dataloader, nonlinear_model, loss_fn, optimizer)

    # checking on the training loss and accuracy once per epoch
    acc, loss = evaluate(train_dataloader, nonlinear_model, loss_fn)
    train_acc_all.append(acc)
    print(f"Epoch {j}: training loss: {loss}, accuracy: {acc}")

    # checking on the validation loss and accuracy once per epoch
    val_acc, val_loss = evaluate(val_dataloader, nonlinear_model, loss_fn)
    val_acc_all.append(val_acc)
    print(f"Epoch {j}: val. loss: {val_loss}, val. accuracy: {val_acc}")
```

    Epoch 0: training loss: 0.8722913052241007, accuracy: 74.675
    Epoch 0: val. loss: 0.8642488090281791, val. accuracy: 75.0
    Epoch 1: training loss: 0.4204843194484711, accuracy: 88.10208333333334
    Epoch 1: val. loss: 0.41196700391617225, val. accuracy: 88.20833333333333
    Epoch 2: training loss: 0.30932906846205394, accuracy: 91.18958333333333
    Epoch 2: val. loss: 0.30337735265493393, val. accuracy: 90.93333333333334
    Epoch 3: training loss: 0.25532810139656065, accuracy: 92.60833333333333
    Epoch 3: val. loss: 0.2523903220574907, val. accuracy: 92.48333333333333
    Epoch 4: training loss: 0.21229399609565736, accuracy: 93.87083333333334
    Epoch 4: val. loss: 0.2120252953089298, val. accuracy: 93.65833333333333
    CPU times: user 36.3 s, sys: 30 s, total: 1min 6s
    Wall time: 12 s

``` python
pltsize=1
plt.figure(figsize=(10*pltsize, 10 * pltsize))
plt.plot(range(epochs), train_acc_all,label = 'Training Acc.' )
plt.plot(range(epochs), val_acc_all, label = 'Validation Acc.' )
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
```

    Text(0.5, 0, 'Epoch #')

    Text(0, 0.5, 'Loss')

![](index_files/figure-commonmark/cell-22-output-3.png)

``` python
show_failures(nonlinear_model, test_dataloader)
```

    Showing max 10 first failures. The predicted class is shown first and the correct class in parentheses.

![](index_files/figure-commonmark/cell-23-output-2.png)

## Recap

To train and validate a neural network model, you need: - Data split
into training/validation/test sets, - A model with parameters to
learn, - An appropriate loss function, - An optimizer (with tunable
parameters such as learning rate, weight decay etc.) used to learn the
parameters of the model.

## Homework

1.  Compare the quality of your model when using different:

- batch sizes,
- learning rates,
- activation functions.

3.  Bonus: What is a learning rate scheduler?

If you have time, experiment with how to improve the model. Note:
training and validation data can be used to compare models, but test
data should be saved until the end as a final check of generalization.

## Homework solution

Make the following changes to the cells with the comment “\#For HW”

``` python
#####################To modify the batch size##########################
batch_size = 32 # 64, 128, 256, 512

# The dataloader makes our dataset iterable 
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size)
val_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
##############################################################################


##########################To change the learning rate##########################
optimizer = torch.optim.SGD(nonlinear_model.parameters(), lr=0.01) #modify the value of lr
##############################################################################


##########################To change activation##########################
###### Go to https://pytorch.org/docs/main/nn.html#non-linear-activations-weighted-sum-nonlinearity for more activations ######
class NonlinearClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers_stack = nn.Sequential(
            nn.Linear(28*28, 50),
            nn.Sigmoid(), #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(50, 50),
            nn.Tanh(), #nn.ReLU(),
           # nn.Dropout(0.2),
            nn.Linear(50, 50),
            nn.ReLU(),
           # nn.Dropout(0.2),
            nn.Linear(50, 10)
        )
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.layers_stack(x)

        return x
##############################################################################
```

Bonus question: A learning rate scheduler is an essential deep learning
technique used to dynamically adjust the learning rate during training.
This strategic can significantly impact the convergence speed and
overall performance of a neural network.See below on how to incorporate
it to your training.

``` python
nonlinear_model = NonlinearClassifier()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(nonlinear_model.parameters(), lr=0.1)

# Step learning rate scheduler: reduce by a factor of 0.1 every 2 epochs (only for illustrative purposes)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
```

``` python
%%time

epochs = 6
train_acc_all = []
val_acc_all = []
for j in range(epochs):
    train_one_epoch(train_dataloader, nonlinear_model, loss_fn, optimizer)
    #step the scheduler
    scheduler.step()

    # Print the current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {j+1}/{epochs}, Learning Rate: {current_lr}")
        
    # checking on the training loss and accuracy once per epoch
    acc, loss = evaluate(train_dataloader, nonlinear_model, loss_fn)
    train_acc_all.append(acc)
    print(f"Epoch {j}: training loss: {loss}, accuracy: {acc}")
    
    # checking on the validation loss and accuracy once per epoch
    val_acc, val_loss = evaluate(val_dataloader, nonlinear_model, loss_fn)
    val_acc_all.append(val_acc)
    print(f"Epoch {j}: val. loss: {val_loss}, val. accuracy: {val_acc}")
```

    Epoch 1/6, Learning Rate: 0.1
    Epoch 0: training loss: 0.3392889190390706, accuracy: 90.18541666666667
    Epoch 0: val. loss: 0.33126659979422884, val. accuracy: 90.35
    Epoch 2/6, Learning Rate: 0.010000000000000002
    Epoch 1: training loss: 0.23930568335577845, accuracy: 92.85833333333333
    Epoch 1: val. loss: 0.23523078672587872, val. accuracy: 92.7
    Epoch 3/6, Learning Rate: 0.010000000000000002
    Epoch 2: training loss: 0.22036956796422602, accuracy: 93.5
    Epoch 2: val. loss: 0.21673137468099593, val. accuracy: 93.35833333333333
    Epoch 4/6, Learning Rate: 0.0010000000000000002
    Epoch 3: training loss: 0.21369351989837984, accuracy: 93.70208333333333
    Epoch 3: val. loss: 0.21026626736919085, val. accuracy: 93.50833333333334
    Epoch 5/6, Learning Rate: 0.0010000000000000002
    Epoch 4: training loss: 0.21219746003051598, accuracy: 93.71041666666666
    Epoch 4: val. loss: 0.20888884751995404, val. accuracy: 93.60000000000001
    Epoch 6/6, Learning Rate: 0.00010000000000000003
    Epoch 5: training loss: 0.21153909859620035, accuracy: 93.71666666666667
    Epoch 5: val. loss: 0.20836285509665808, val. accuracy: 93.55833333333334
    CPU times: user 46.2 s, sys: 44.5 s, total: 1min 30s
    Wall time: 18.6 s
