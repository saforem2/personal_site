# Convolutional Neural Networks
Sam Foreman, Huihuo Zheng, Corey Adams, Bethany Lusch

<link rel="preconnect" href="https://fonts.googleapis.com">

- [Convolutional Networks: A brief historical
  context](#convolutional-networks-a-brief-historical-context)
- [Convolutional Building Blocks](#convolutional-building-blocks)
  - [Convolutions](#convolutions)
  - [Normalization](#normalization)
  - [Downsampling (And upsampling)](#downsampling-and-upsampling)
  - [Residual Connections](#residual-connections)
- [Building a ConvNet](#building-a-convnet)
  - [Homework 1:](#homework-1)

Up until transformers, convolutions were *the* state of the art in
computer vision.  
In many ways and applications they still are!

Large Language Models, which are what we’ll focus on the rest of the
series after this lecture, are really good at ordered, \*tokenized data.
But there is lots of data that isn’t *implicitly* ordered like `images`,
and their more general cousins `graphs`.

Today’s lecture focuses on computer vision models, and particularly on
convolutional neural networks. There are a ton of applications you can
do with these, and not nearly enough time to get into them. Check out
the extra references file to see some publications to get you started if
you want to learn more.

Tip: this notebook is much faster on the GPU!

## Convolutional Networks: A brief historical context

![ImageNet Accuracy by Yearh](./ImageNet.png)

[reference](https://www.researchgate.net/publication/332452649_A_Roadmap_for_Foundational_Research_on_Artificial_Intelligence_in_Medical_Imaging_From_the_2018_NIHRSNAACRThe_Academy_Workshop)

``` python
import torch, torchvision
```

# Convolutional Building Blocks

We’re going to go through some examples of building blocks for
convolutional networks. To help illustate some of these, let’s use an
image for examples:

``` python
from PIL import Image
# wget line useful in Google Colab
! wget https://raw.githubusercontent.com/argonne-lcf/ai-science-training-series/main/03_advanced_neural_networks/ALCF-Staff.jpg
alcf_image = Image.open("ALCF-Staff.jpg")
```

    --2025-07-17 15:58:04--  https://raw.githubusercontent.com/argonne-lcf/ai-science-training-series/main/03_advanced_neural_networks/ALCF-Staff.jpg
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.110.133, 185.199.111.133, 185.199.108.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.110.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 417835 (408K) [image/jpeg]
    Saving to: ‘ALCF-Staff.jpg.12’

    ALCF-Staff.jpg.12     0%[                    ]       0  --.-KB/s               ALCF-Staff.jpg.12   100%[===================>] 408.04K  --.-KB/s    in 0.02s   

    2025-07-17 15:58:04 (25.9 MB/s) - ‘ALCF-Staff.jpg.12’ saved [417835/417835]

``` python
from matplotlib import pyplot as plt
figure = plt.figure(figsize=(20,20))
plt.imshow(alcf_image)
plt.show()
```

![](index_files/figure-commonmark/cell-4-output-1.png)

## Convolutions

Convolutions are a restriction of - and a specialization of - dense
linear layers. A convolution of an image produces another image, and
each output pixel is a function of only it’s local neighborhood of
points. This is called an *inductive bias* and is a big reason why
convolutions work for image data: neighboring pixels are correlated and
you can operate on just those pixels at a time.

See examples of convolutions
[here](https://github.com/vdumoulin/conv_arithmetic)

![image-2.png](./conv_eqn.png)

![image.png](./conv.png)

``` python
# Let's apply a convolution to the ALCF Staff photo:
alcf_tensor = torchvision.transforms.ToTensor()(alcf_image)

# Reshape the tensor to have a batch size of 1:
alcf_tensor = alcf_tensor.reshape((1,) + alcf_tensor.shape)

# Create a random convolution:
# shape is: (channels_in, channels_out, kernel_x, kernel_y)
conv_random = torch.rand((3,3,15,15))

alcf_rand = torch.nn.functional.conv2d(alcf_tensor, conv_random)
alcf_rand = (1./alcf_rand.max()) * alcf_rand
print(alcf_rand.shape)
alcf_rand = alcf_rand.reshape(alcf_rand.shape[1:])

print(alcf_tensor.shape)

rand_image = alcf_rand.permute((1,2,0)).cpu()

figure = plt.figure(figsize=(20,20))

plt.imshow(rand_image)
```

    torch.Size([1, 3, 1111, 1986])
    torch.Size([1, 3, 1125, 2000])

![](index_files/figure-commonmark/cell-5-output-2.png)

## Normalization

![Batch Norm](./batch_norm.png) Reference:
[Normalizations](https://arxiv.org/pdf/1903.10520.pdf)

Normalization is the act of transforming the mean and moment of your
data to standard values (usually 0.0 and 1.0). It’s particularly useful
in machine learning since it stabilizes training, and allows higher
learning rates.

![Batch Normalization accelerates training](./batch_norm_effect.png)

Reference: [Batch Norm](https://arxiv.org/pdf/1502.03167.pdf)

``` python
# Let's apply a normalization to the ALCF Staff photo:
alcf_tensor = torchvision.transforms.ToTensor()(alcf_image)

# Reshape the tensor to have a batch size of 1:
alcf_tensor = alcf_tensor.reshape((1,) + alcf_tensor.shape)


alcf_rand = torch.nn.functional.normalize(alcf_tensor)
alcf_rand = alcf_rand.reshape(alcf_rand.shape[1:])

print(alcf_tensor.shape)

rand_image = alcf_rand.permute((1,2,0)).cpu()

figure = plt.figure(figsize=(20,20))

plt.imshow(rand_image)
```

    torch.Size([1, 3, 1125, 2000])

![](index_files/figure-commonmark/cell-6-output-2.png)

## Downsampling (And upsampling)

Downsampling is a critical component of convolutional and many vision
models. Because of the local-only nature of convolutional filters,
learning large-range features can be too slow for convergence.
Downsampling of layers can bring information from far away closer,
effectively changing what it means to be “local” as the input to a
convolution.

![Convolutional Pooling](./conv_pooling.png)

[Reference](https://www.researchgate.net/publication/333593451_Application_of_Transfer_Learning_Using_Convolutional_Neural_Network_Method_for_Early_Detection_of_Terry's_Nail)

``` python
# Let's apply a normalization to the ALCF Staff photo:
alcf_tensor = torchvision.transforms.ToTensor()(alcf_image)

# Reshape the tensor to have a batch size of 1:
alcf_tensor = alcf_tensor.reshape((1,) + alcf_tensor.shape)


alcf_rand = torch.nn.functional.max_pool2d(alcf_tensor, 2)
alcf_rand = alcf_rand.reshape(alcf_rand.shape[1:])

print(alcf_tensor.shape)

rand_image = alcf_rand.permute((1,2,0)).cpu()

figure = plt.figure(figsize=(20,20))

plt.imshow(rand_image)
```

    torch.Size([1, 3, 1125, 2000])

![](index_files/figure-commonmark/cell-7-output-2.png)

## Residual Connections

One issue, quickly encountered when making convolutional networks deeper
and deeper, is the “Vanishing Gradients” problem. As layers were stacked
on top of each other, the size of updates dimished at the earlier layers
of a convolutional network. The paper “Deep Residual Learning for Image
Recognition” solved this by introduction “residual connections” as skip
layers.

Reference: [Deep Residual Learning for Image
Recognition](https://arxiv.org/pdf/1512.03385.pdf)

![Residual Layer](./residual_layer.png)

Compare the performance of the models before and after the introduction
of these layers:

![Resnet Performance vs. Plain network
performance](./resnet_comparison.png)

If you have time to read only one paper on computer vision, make it this
one! Resnet was the first model to beat human accuracy on ImageNet and
is one of the most impactful papers in AI ever published.

# Building a ConvNet

In this section we’ll build and apply a conv net to the mnist dataset.
The layers here are loosely based off of the ConvNext architecture. Why?
Because we’re getting into LLM’s soon, and this ConvNet uses LLM
features. ConvNext is an update to the ResNet architecture that
outperforms it.

[ConvNext](https://arxiv.org/abs/2201.03545)

The dataset here is CIFAR-10 - slightly harder than MNIST but still
relatively easy and computationally tractable.

``` python
from torchvision.transforms import v2
training_data = torchvision.datasets.CIFAR10(
    # Polaris: root="/lus/eagle/projects/datasets/CIFAR-10/",
    # Polaris: download=False,
    root="data",
    download=True,
    train=True,
    transform=v2.Compose([
        v2.ToTensor(),
        v2.RandomHorizontalFlip(),
        v2.RandomResizedCrop(size=32, scale=[0.85,1.0], antialias=False),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    ])
)

test_data = torchvision.datasets.CIFAR10(
    # Polaris: root="/lus/eagle/projects/datasets/CIFAR-10/",
    # Polaris: download=False,
    root="data",
    download=True,
    train=False,
    transform=torchvision.transforms.ToTensor()
)

training_data, validation_data = torch.utils.data.random_split(training_data, [0.8, 0.2], generator=torch.Generator().manual_seed(55))

batch_size = 128

# The dataloader makes our dataset iterable 
train_dataloader = torch.utils.data.DataLoader(training_data, 
    batch_size=batch_size, 
    pin_memory=True,
    shuffle=True, 
    num_workers=2)
val_dataloader = torch.utils.data.DataLoader(validation_data, 
    batch_size=batch_size, 
    pin_memory=True,
    shuffle=False, 
    num_workers=2)
```

    /Users/samforeman/projects/saforem2/personal_site_CLEAN/.venv/lib/python3.12/site-packages/torchvision/transforms/v2/_deprecated.py:42: UserWarning: The transform `ToTensor()` is deprecated and will be removed in a future release. Instead, please use `v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])`.Output is equivalent up to float precision.
      warnings.warn(

``` python
from matplotlib import pyplot as plt
%matplotlib inline
```

``` python
batch, (X, Y) = next(enumerate(train_dataloader))
plt.imshow(X[0].cpu().permute((1,2,0))); plt.show()
```

![](index_files/figure-commonmark/cell-10-output-1.png)

This code below is important as our models get bigger: this is wrapping
the pytorch data loaders to put the data onto the GPU!

``` python
dev = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


def preprocess(x, y):
    # CIFAR-10 is *color* images so 3 layers!
    return x.view(-1, 3, 32, 32).to(dev), y.to(dev)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))


train_dataloader = WrappedDataLoader(train_dataloader, preprocess)
val_dataloader = WrappedDataLoader(val_dataloader, preprocess)
```

``` python
from torch import nn


class Downsampler(nn.Module):

    def __init__(self, in_channels, out_channels, shape, stride=2):
        super(Downsampler, self).__init__()

        self.norm = nn.LayerNorm([in_channels, *shape])

        self.downsample = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size = stride,
            stride = stride,
        )

    def forward(self, inputs):


        return self.downsample(self.norm(inputs))



class ConvNextBlock(nn.Module):
    """This block of operations is loosely based on this paper:

    """


    def __init__(self, in_channels, shape):
        super(ConvNextBlock, self).__init__()

        # Depthwise, seperable convolution with a large number of output filters:
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                                     out_channels=in_channels, 
                                     groups=in_channels,
                                     kernel_size=[7,7],
                                     padding='same' )

        self.norm = nn.LayerNorm([in_channels, *shape])

        # Two more convolutions:
        self.conv2 = nn.Conv2d(in_channels=in_channels, 
                                     out_channels=4*in_channels,
                                     kernel_size=1)

        self.conv3 = nn.Conv2d(in_channels=4*in_channels, 
                                     out_channels=in_channels,
                                     kernel_size=1
                                     )


    def forward(self, inputs):
        x = self.conv1(inputs)

        # The normalization layer:
        x = self.norm(x)

        x = self.conv2(x)

        # The non-linear activation layer:
        x = torch.nn.functional.gelu(x)

        x = self.conv3(x)

        # This makes it a residual network:
        return x + inputs


class Classifier(nn.Module):


    def __init__(self, n_initial_filters, n_stages, blocks_per_stage):
        super(Classifier, self).__init__()

        # This is a downsampling convolution that will produce patches of output.

        # This is similar to what vision transformers do to tokenize the images.
        self.stem = nn.Conv2d(in_channels=3,
                                    out_channels=n_initial_filters,
                                    kernel_size=1,
                                    stride=1)

        current_shape = [32, 32]

        self.norm1 = nn.LayerNorm([n_initial_filters,*current_shape])
        # self.norm1 = WrappedLayerNorm()

        current_n_filters = n_initial_filters

        self.layers = nn.Sequential()
        for i, n_blocks in enumerate(range(n_stages)):
            # Add a convnext block series:
            for _ in range(blocks_per_stage):
                self.layers.append(ConvNextBlock(in_channels=current_n_filters, shape=current_shape))
            # Add a downsampling layer:
            if i != n_stages - 1:
                # Skip downsampling if it's the last layer!
                self.layers.append(Downsampler(
                    in_channels=current_n_filters, 
                    out_channels=2*current_n_filters,
                    shape = current_shape,
                    )
                )
                # Double the number of filters:
                current_n_filters = 2*current_n_filters
                # Cut the shape in half:
                current_shape = [ cs // 2 for cs in current_shape]



        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(current_n_filters),
            nn.Linear(current_n_filters, 10)
        )
        # self.norm2 = nn.InstanceNorm2d(current_n_filters)
        # # This brings it down to one channel / class
        # self.bottleneck = nn.Conv2d(in_channels=current_n_filters, out_channels=10, 
        #                                   kernel_size=1, stride=1)

    def forward(self, inputs):

        x = self.stem(inputs)
        # Apply a normalization after the initial patching:
        x = self.norm1(x)

        # Apply the main chunk of the network:
        x = self.layers(x)

        # Normalize and readout:
        x = nn.functional.avg_pool2d(x, x.shape[2:])
        x = self.head(x)

        return x



        # x = self.norm2(x)
        # x = self.bottleneck(x)

        # # Average pooling of the remaining spatial dimensions (and reshape) makes this label-like:
        # return nn.functional.avg_pool2d(x, kernel_size=x.shape[-2:]).reshape((-1,10))
```

``` python
!pip install torchinfo # if not on Polaris
```

    Requirement already satisfied: torchinfo in /opt/homebrew/lib/python3.11/site-packages (1.8.0)

``` python
model = Classifier(32, 4, 2).to(device=dev)

from torchinfo import summary

print(summary(model, input_size=(batch_size, 3, 32, 32)))
```

    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Classifier                               [128, 10]                 --
    ├─Conv2d: 1-1                            [128, 32, 32, 32]         128
    ├─LayerNorm: 1-2                         [128, 32, 32, 32]         65,536
    ├─Sequential: 1-3                        [128, 256, 4, 4]          --
    │    └─ConvNextBlock: 2-1                [128, 32, 32, 32]         --
    │    │    └─Conv2d: 3-1                  [128, 32, 32, 32]         1,600
    │    │    └─LayerNorm: 3-2               [128, 32, 32, 32]         65,536
    │    │    └─Conv2d: 3-3                  [128, 128, 32, 32]        4,224
    │    │    └─Conv2d: 3-4                  [128, 32, 32, 32]         4,128
    │    └─ConvNextBlock: 2-2                [128, 32, 32, 32]         --
    │    │    └─Conv2d: 3-5                  [128, 32, 32, 32]         1,600
    │    │    └─LayerNorm: 3-6               [128, 32, 32, 32]         65,536
    │    │    └─Conv2d: 3-7                  [128, 128, 32, 32]        4,224
    │    │    └─Conv2d: 3-8                  [128, 32, 32, 32]         4,128
    │    └─Downsampler: 2-3                  [128, 64, 16, 16]         --
    │    │    └─LayerNorm: 3-9               [128, 32, 32, 32]         65,536
    │    │    └─Conv2d: 3-10                 [128, 64, 16, 16]         8,256
    │    └─ConvNextBlock: 2-4                [128, 64, 16, 16]         --
    │    │    └─Conv2d: 3-11                 [128, 64, 16, 16]         3,200
    │    │    └─LayerNorm: 3-12              [128, 64, 16, 16]         32,768
    │    │    └─Conv2d: 3-13                 [128, 256, 16, 16]        16,640
    │    │    └─Conv2d: 3-14                 [128, 64, 16, 16]         16,448
    │    └─ConvNextBlock: 2-5                [128, 64, 16, 16]         --
    │    │    └─Conv2d: 3-15                 [128, 64, 16, 16]         3,200
    │    │    └─LayerNorm: 3-16              [128, 64, 16, 16]         32,768
    │    │    └─Conv2d: 3-17                 [128, 256, 16, 16]        16,640
    │    │    └─Conv2d: 3-18                 [128, 64, 16, 16]         16,448
    │    └─Downsampler: 2-6                  [128, 128, 8, 8]          --
    │    │    └─LayerNorm: 3-19              [128, 64, 16, 16]         32,768
    │    │    └─Conv2d: 3-20                 [128, 128, 8, 8]          32,896
    │    └─ConvNextBlock: 2-7                [128, 128, 8, 8]          --
    │    │    └─Conv2d: 3-21                 [128, 128, 8, 8]          6,400
    │    │    └─LayerNorm: 3-22              [128, 128, 8, 8]          16,384
    │    │    └─Conv2d: 3-23                 [128, 512, 8, 8]          66,048
    │    │    └─Conv2d: 3-24                 [128, 128, 8, 8]          65,664
    │    └─ConvNextBlock: 2-8                [128, 128, 8, 8]          --
    │    │    └─Conv2d: 3-25                 [128, 128, 8, 8]          6,400
    │    │    └─LayerNorm: 3-26              [128, 128, 8, 8]          16,384
    │    │    └─Conv2d: 3-27                 [128, 512, 8, 8]          66,048
    │    │    └─Conv2d: 3-28                 [128, 128, 8, 8]          65,664
    │    └─Downsampler: 2-9                  [128, 256, 4, 4]          --
    │    │    └─LayerNorm: 3-29              [128, 128, 8, 8]          16,384
    │    │    └─Conv2d: 3-30                 [128, 256, 4, 4]          131,328
    │    └─ConvNextBlock: 2-10               [128, 256, 4, 4]          --
    │    │    └─Conv2d: 3-31                 [128, 256, 4, 4]          12,800
    │    │    └─LayerNorm: 3-32              [128, 256, 4, 4]          8,192
    │    │    └─Conv2d: 3-33                 [128, 1024, 4, 4]         263,168
    │    │    └─Conv2d: 3-34                 [128, 256, 4, 4]          262,400
    │    └─ConvNextBlock: 2-11               [128, 256, 4, 4]          --
    │    │    └─Conv2d: 3-35                 [128, 256, 4, 4]          12,800
    │    │    └─LayerNorm: 3-36              [128, 256, 4, 4]          8,192
    │    │    └─Conv2d: 3-37                 [128, 1024, 4, 4]         263,168
    │    │    └─Conv2d: 3-38                 [128, 256, 4, 4]          262,400
    ├─Sequential: 1-4                        [128, 10]                 --
    │    └─Flatten: 2-12                     [128, 256]                --
    │    └─LayerNorm: 2-13                   [128, 256]                512
    │    └─Linear: 2-14                      [128, 10]                 2,570
    ==========================================================================================
    Total params: 2,047,114
    Trainable params: 2,047,114
    Non-trainable params: 0
    Total mult-adds (Units.GIGABYTES): 10.34
    ==========================================================================================
    Input size (MB): 1.57
    Forward/backward pass size (MB): 1036.27
    Params size (MB): 8.19
    Estimated Total Size (MB): 1046.03
    ==========================================================================================

``` python
def evaluate(dataloader, model, loss_fn, val_bar):
    # Set the model to evaluation mode - some NN pieces behave differently during training
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader)
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
            val_bar.update()

    loss /= num_batches
    correct /= (size*batch_size)

    accuracy = 100*correct
    return accuracy, loss
```

``` python
def train_one_epoch(dataloader, model, loss_fn, optimizer, progress_bar):
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

        progress_bar.update()
```

``` python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.01)
```

``` python
import time
import ezpz
from tqdm.notebook import tqdm

def train_step(x, y):
    t0 = time.perf_counter()
    # Forward pass
    pred = model(x)
    loss = loss_fn(pred, y)
    t1 = time.perf_counter()

    # Backward pass
    loss.backward()
    t2 = time.perf_counter()

    # Update weights
    optimizer.step()
    t3 = time.perf_counter()

    # Reset gradients
    optimizer.zero_grad()
    t4 = time.perf_counter()

    return loss.item(), {
        "dtf": t1 - t0,
        "dtb": t2 - t1,
        "dtu": t3 - t2,
        "dtz": t4 - t3,
    }

logger = ezpz.get_logger("3-conv-nets")
history = ezpz.History()
for i in range(10):
    t0 = time.perf_counter()
    x, y = next(iter(train_dataloader))
    t1 = time.perf_counter()
    loss, dt = train_step(x, y)
    logger.info(
        history.update(
            {
                "iter": i,
                "loss": loss,
                "dtd": t1 - t0,
                **dt,
            }
        )
    )



# epochs = 1
# for j in range(epochs):
#     with tqdm(total=len(train_dataloader), position=0, leave=True, desc=f"Train Epoch {j}") as train_bar:
#         train_one_epoch(train_dataloader, model, loss_fn, optimizer, train_bar)
#
#     # checking on the training & validation loss & accuracy 
#     # for training data - only once every 5 epochs (takes a while) 
#     if j % 5 == 0:
#         with tqdm(total=len(train_dataloader), position=0, leave=True, desc=f"Validate (train) Epoch {j}") as train_eval:
#             acc, loss = evaluate(train_dataloader, model, loss_fn, train_eval)
#             print(f"Epoch {j}: training loss: {loss:.3f}, accuracy: {acc:.3f}")
#
#     with tqdm(total=len(val_dataloader), position=0, leave=True, desc=f"Validate Epoch {j}") as val_bar:
#         acc_val, loss_val = evaluate(val_dataloader, model, loss_fn, val_bar)
#         print(f"Epoch {j}: validation loss: {loss_val:.3f}, accuracy: {acc_val:.3f}")
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-07-17 </span><span style="color: #808080; text-decoration-color: #808080">15:58:40</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_56591</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3332296832</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">38</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">3-conv-nets</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">iter</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2.373971</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtd</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">4.278633</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.797844</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">13.322712</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtu</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.003188</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtz</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.000984</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-07-17 </span><span style="color: #808080; text-decoration-color: #808080">15:59:01</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_56591</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3332296832</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">38</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">3-conv-nets</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">iter</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">1</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2.392965</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtd</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">5.155691</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.833662</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">15.278990</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtu</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.003590</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtz</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.000597</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-07-17 </span><span style="color: #808080; text-decoration-color: #808080">15:59:21</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_56591</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3332296832</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">38</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">3-conv-nets</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">iter</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2.382057</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtd</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">4.545751</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.832608</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">14.970146</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtu</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.004192</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtz</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.000399</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-07-17 </span><span style="color: #808080; text-decoration-color: #808080">15:59:41</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_56591</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3332296832</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">38</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">3-conv-nets</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">iter</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">3</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2.332808</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtd</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">4.976565</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.860296</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">13.959051</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtu</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.002798</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtz</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.000835</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-07-17 </span><span style="color: #808080; text-decoration-color: #808080">16:00:01</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_56591</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3332296832</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">38</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">3-conv-nets</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">iter</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">4</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2.311733</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtd</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">4.749009</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.832350</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">14.573133</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtu</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.004266</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtz</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.000927</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-07-17 </span><span style="color: #808080; text-decoration-color: #808080">16:00:23</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_56591</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3332296832</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">38</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">3-conv-nets</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">iter</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">5</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2.317462</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtd</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">5.237694</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.819806</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">15.844323</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtu</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.005773</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtz</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.000734</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-07-17 </span><span style="color: #808080; text-decoration-color: #808080">16:00:43</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_56591</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3332296832</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">38</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">3-conv-nets</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">iter</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">6</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2.345009</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtd</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">4.664517</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.847006</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">14.345617</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtu</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.003211</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtz</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.000806</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-07-17 </span><span style="color: #808080; text-decoration-color: #808080">16:01:04</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_56591</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3332296832</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">38</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">3-conv-nets</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">iter</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">7</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2.334987</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtd</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">5.072934</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.799330</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">14.638548</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtu</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.003020</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtz</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.000588</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-07-17 </span><span style="color: #808080; text-decoration-color: #808080">16:01:23</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_56591</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3332296832</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">38</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">3-conv-nets</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">iter</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">8</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2.301219</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtd</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">4.546424</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.791437</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">14.118925</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtu</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.003136</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtz</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.000909</span>
</pre>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #000000; text-decoration-color: #000000">[</span><span style="color: #000000; text-decoration-color: #000000">2025-07-17 </span><span style="color: #808080; text-decoration-color: #808080">16:01:43</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008000; text-decoration-color: #008000; font-weight: bold">I</span><span style="color: #000000; text-decoration-color: #000000">][</span><span style="color: #008080; text-decoration-color: #008080; font-style: italic">ipykernel_56591</span><span style="color: #000000; text-decoration-color: #000000">/</span><span style="color: #000080; text-decoration-color: #000080">3332296832</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #000000; text-decoration-color: #000000">38</span><span style="color: #0000ff; text-decoration-color: #0000ff">:</span><span style="color: #00ff00; text-decoration-color: #00ff00; font-style: italic">3-conv-nets</span><span style="color: #000000; text-decoration-color: #000000">]</span><span style="color: #838383; text-decoration-color: #838383"> </span><span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">iter</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">9</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">loss</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">2.315814</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtd</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">4.968656</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtf</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.968394</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtb</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">14.204992</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtu</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.004923</span> <span style="color: #0000ff; text-decoration-color: #0000ff; font-style: italic">dtz</span>=<span style="color: #ff00ff; text-decoration-color: #ff00ff">0.001021</span>
</pre>

## Homework 1:

In this notebook, we’ve learned about some basic convolutional networks
and trained one on CIFAR-10 images. It did … OK. There is significant
overfitting of this model. There are some ways to address that, but we
didn’t have time to get into that in this session.

Meanwhile, your homework (part 1) for this week is to try to train the
model again but with a different architecture. Change one or more of the
following: - The number of convolutions between downsampling - The
number of filters in each layer - The initial “patchify” layer - Another
hyper-parameter of your choosing

And compare your final validation accuracy to the accuracy shown here.
Can you beat the validation accuracy shown?

For full credit on the homework, you need to show (via text, or make a
plot) the training and validation data sets’ performance (loss and
accuracy) for all the epochs you train. You also need to explain, in
several sentences, what you changed in the network and why you think it
makes a difference.
