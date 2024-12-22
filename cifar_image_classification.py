# In this problem, we will explore different deep learning architectures for image classification on the CIFAR-10 dataset.
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)  # this should print out CUDA

import torch
from torch import nn
import numpy as np

from typing import Tuple, Union, List, Callable
from torch.optim import SGD
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

# %matplotlib inline

# Check if we are using GPU, if it is available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)  # this should print out CUDA

#load the CIFAR-10 data
train_dataset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=torchvision.transforms.ToTensor())

SAMPLE_DATA = False # set this to True if you want to speed up training when searching for hyperparameters!

batch_size = 128

if SAMPLE_DATA:
  train_dataset, _ = random_split(train_dataset, [int(0.1 * len(train_dataset)), int(0.9 * len(train_dataset))]) # get 10% of train dataset and "throw away" the other 90%

train_dataset, val_dataset = random_split(train_dataset, [int(0.9 * len(train_dataset)), int( 0.1 * len(train_dataset))])

# Create separate dataloaders for the train, test, and validation set
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=True
)


# Prints out elements from the data
imgs, labels = next(iter(train_loader))
print(f"A single batch of images has shape: {imgs.size()}")
example_image, example_label = imgs[0], labels[0]
c, w, h = example_image.size()
print(f"A single RGB image has {c} channels, width {w}, and height {h}.")

# This is one way to flatten our images
batch_flat_view = imgs.view(-1, c * w * h)
print(f"Size of a batch of images flattened with view: {batch_flat_view.size()}")

# This is another equivalent way
batch_flat_flatten = imgs.flatten(1)
print(f"Size of a batch of images flattened with flatten: {batch_flat_flatten.size()}")

# The new dimension is just the product of the ones we flattened
d = example_image.flatten().size()[0]
print(c * w * h == d)

# View the image
t =  torchvision.transforms.ToPILImage()
plt.imshow(t(example_image))

# These are what the class labels in CIFAR-10 represent. For more information,
# visit https://www.cs.toronto.edu/~kriz/cifar.html
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
           "horse", "ship", "truck"]
print(f"This image is labeled as class {classes[example_label]}")

# Linear Model
def linear_model() -> nn.Module:
    """Instantiate a linear model and send it to device."""
    model =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(d, 10)
         )
    return model.to(DEVICE)

def train(
    model: nn.Module, optimizer: SGD,
    train_loader: DataLoader, val_loader: DataLoader,
    epochs: int = 20
    )-> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Trains a model for the specified number of epochs using the loaders.

    Returns:
    Lists of training loss, training accuracy, validation loss, validation accuracy for each epoch.
    """

    loss = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    for e in tqdm(range(epochs)):
        model.train()
        train_loss = 0.0
        train_acc = 0.0

        # Main training loop; iterate over train_loader. The loop
        # terminates when the train loader finishes iterating, which is one epoch.
        for (x_batch, labels) in train_loader:
            x_batch, labels = x_batch.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            labels_pred = model(x_batch)
            batch_loss = loss(labels_pred, labels)
            train_loss = train_loss + batch_loss.item()

            labels_pred_max = torch.argmax(labels_pred, 1)
            batch_acc = torch.sum(labels_pred_max == labels)
            train_acc = train_acc + batch_acc.item()

            batch_loss.backward()
            optimizer.step()
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc / (batch_size * len(train_loader)))

        # Validation loop; use .no_grad() context manager to save memory.
        model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.no_grad():
            for (v_batch, labels) in val_loader:
                v_batch, labels = v_batch.to(DEVICE), labels.to(DEVICE)
                labels_pred = model(v_batch)
                v_batch_loss = loss(labels_pred, labels)
                val_loss = val_loss + v_batch_loss.item()

                v_pred_max = torch.argmax(labels_pred, 1)
                batch_acc = torch.sum(v_pred_max == labels)
                val_acc = val_acc + batch_acc.item()
            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_acc / (batch_size * len(val_loader)))

    return train_losses, train_accuracies, val_losses, val_accuracies

def parameter_search(train_loader: DataLoader,
                     val_loader: DataLoader,
                     model_fn:Callable[[], nn.Module]) -> float:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The learning rate with the least validation loss.
    """
    num_iter = 10
    best_loss = torch.tensor(np.inf)
    best_lr = 0.0

    lrs = torch.linspace(10 ** (-6), 10 ** (-1), num_iter)

    for lr in lrs:
        print(f"trying learning rate {lr}")
        model = model_fn()
        optim = SGD(model.parameters(), lr)
        train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=20
            )

        if min(val_loss) < best_loss:
            best_loss = min(val_loss)
            best_lr = lr

    return best_lr

# Train and evaluate linear model
best_lr = parameter_search(train_loader, val_loader, linear_model)

model = linear_model()
optimizer = SGD(model.parameters(), best_lr)

train_loss, train_accuracy, val_loss, val_accuracy = train(
    model, optimizer, train_loader, val_loader, 20)

# Create plot of training and validation accuracy for each epoch
epochs = range(1, 21)
plt.plot(epochs, train_accuracy, label="Train Accuracy")
plt.plot(epochs, val_accuracy, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Logistic Regression Accuracy for CIFAR-10 vs Epoch")
plt.show()

# Evaluate model on testing data
def evaluate(
    model: nn.Module, loader: DataLoader
) -> Tuple[float, float]:
    """Computes test loss and accuracy of model on loader."""
    loss = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for (batch, labels) in loader:
            batch, labels = batch.to(DEVICE), labels.to(DEVICE)
            y_batch_pred = model(batch)
            batch_loss = loss(y_batch_pred, labels)
            test_loss = test_loss + batch_loss.item()

            pred_max = torch.argmax(y_batch_pred, 1)
            batch_acc = torch.sum(pred_max == labels)
            test_acc = test_acc + batch_acc.item()
        test_loss = test_loss / len(loader)
        test_acc = test_acc / (batch_size * len(loader))
        return test_loss, test_acc

test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Accuracy: {test_acc}")

# Fully Connected layer implementation
import random
def fully_connected_layer(M) -> nn.Module:
    model = nn.Sequential(nn.Flatten(), nn.Linear(d, M), nn.ReLU(), nn.Linear(M, 10))
    return model.to(DEVICE)

#Training
#use the same training method from before

#Search for Parameters
def fully_connected_parameter_search(train_loader: DataLoader,
                     val_loader: DataLoader,
                     model_fn:Callable[[], nn.Module]) -> float:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The learning rate with the least validation loss.
    """
    num_iter = 10
    best_loss = torch.tensor(np.inf)
    best_lr = 0.0
    best_M = 0

    lrs = torch.linspace(10 ** (-6), 10 ** (-1), num_iter)

    for _ in range(num_iter):
        lr = random.choice(lrs)
        hidden_layer = random.randint(100, 999)

        print(f"trying learning rate {lr} and M {hidden_layer}")
        model = model_fn(hidden_layer)
        optim = SGD(model.parameters(), lr)
        train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=20
            )

        if min(val_loss) < best_loss:
            best_loss = min(val_loss)
            best_lr = lr
            best_M = hidden_layer

    return best_lr, best_M

#find best learning rate and hidden layer size
best_lr, best_M = fully_connected_parameter_search(train_loader, val_loader, fully_connected_layer)

"""Code below is for training on the CPU on colab if GPU isn't available, significantly slower but more automated"""
# #For CPU training
# iterations = 10
# true_best_lr = -1
# true_best_M = -1
# best_acc = -1
# stored_info = {}
# iter = 1
# while best_acc < 0.5:
#     current_info = {}
#     best_lr, best_M = fully_connected_parameter_search(train_loader_small, val_loader_small, fully_connected_layer)
#     current_info["lr"] = best_lr
#     current_info["M"] = best_M

#     model = fully_connected_layer(best_M)
#     optimizer = SGD(model.parameters(), best_lr)
#     train_loss, train_accuracy, val_loss, val_accuracy = train(
#         model, optimizer, train_loader, val_loader, 20)
#     test_loss, test_acc = evaluate(model, test_loader_small)
#     current_info["accuracy"] = test_acc
#     print(f"\nCurrent Iteration: {iter},\nCurrent Learning Rate: {best_lr},\nCurrent M: {best_M},\nCurrent Test Accuracy: {test_acc}")

#     if test_acc > best_acc:
#         if "best" not in stored_info.keys():
#             stored_info["best"] = {}
#         else:
#             previous_best = stored_info["best"]["accuracy"]
#             previous_iteration = stored_info["best"]["iteration"]
#             print(f"\nPrevious Best Accuracy: {previous_best} from iteration {previous_iteration}")
#         print(f"Best Accuracy: {test_acc},\nBest lr: {best_lr},\nBest M: {best_M}")
#         best_acc = test_acc
#         true_best_lr = best_lr
#         true_best_M = best_M
#         stored_info["best"]["accuracy"] = best_acc
#         stored_info["best"]["lr"] = best_lr
#         stored_info["best"]["M"] = best_M
#         stored_info["best"]["iteration"] = iter
#     stored_info[iter] = current_info
#     iter += 1
#     print("\n\n\n")
# print(true_best_lr, true_best_M, best_acc)

print(f"best learning rate: {best_lr} and best M: {best_M}")
model = fully_connected_layer(best_M)
optimizer = SGD(model.parameters(), best_lr)

train_loss, train_accuracy, val_loss, val_accuracy = train(
    model, optimizer, train_loader, val_loader, 20)

epochs = range(1, 21)
plt.plot(epochs, train_accuracy, label="Train Accuracy")
plt.plot(epochs, val_accuracy, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Logistic Regression Accuracy for CIFAR-10 vs Epoch")
plt.show()

test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Accuracy: {test_acc}")

#Make plots for 3 best hyperparameter pairs found during testing
plt.figure(figsize=(12, 6))

#graph 1 (best)
best_lr, best_M = 0.06666699796915054, 999
model = fully_connected_layer(best_M)
optimizer = SGD(model.parameters(), lr=best_lr)
train_loss_1, train_accuracy_1, val_loss_1, val_accuracy_1 = train(
    model, optimizer, train_loader, val_loader, 20)
plt.plot(range(1, 21), train_accuracy_1, label="train accuracy: best learning rate: 0.06666699796915054 and best M: 999")
plt.plot(range(1, 21), val_accuracy_1, label="val accuracy: best learning rate: 0.06666699796915054 and best M: 999", linestyle='dashed')

#graph 2
best_lr, best_M = 0.08888900279998779, 671
model = fully_connected_layer(best_M)
optimizer = SGD(model.parameters(), lr=best_lr)
train_loss_2, train_accuracy_2, val_loss_2, val_accuracy_2 = train(
    model, optimizer, train_loader, val_loader, 20)
plt.plot(range(1, 21), train_accuracy_2, label="train accuracy: best learning rate: 0.08888900279998779 and best M: 671")
plt.plot(val_accuracy_2, label="val accuracy: best learning rate: 0.08888900279998779 and best M: 671", linestyle='dashed')


#graph 3
best_lr, best_M = 0.05555599927902222, 828
model = fully_connected_layer(best_M)
optimizer = SGD(model.parameters(), lr=best_lr)
train_loss_3, train_accuracy_3, val_loss_3, val_accuracy_3 = train(
    model, optimizer, train_loader, val_loader, 20)
plt.plot(range(1, 21), train_accuracy_3, label="train accuracy: best learning rate: 0.05555599927902222 and best M: 828")
plt.plot(range(1, 21), val_accuracy_3, label="val accuracy: best learning rate: 0.05555599927902222 and best M: 828", linestyle='dashed')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.axhline(0.5, linestyle='--')
plt.grid('on')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.show()


#Convolutional Neural Network
#parameters M, k, N, learning rate, and momentum are hyperparameters
#M = number of filters
#each filter is of size k x k x 3
#the max pool is of size N x N
import math
def convolutional_model(M, k, N) -> nn.Module:
    model = nn.Sequential(nn.Conv2d(in_channels=3,
                                    out_channels=M,
                                    kernel_size=k,
                                    stride=1,
                                    padding=0),
                          nn.ReLU(),
                          nn.MaxPool2d(N),
                          nn.Flatten(),
                          nn.Linear(M * (math.floor((33 - k) / N) ** 2), 10))
    return model.to(DEVICE)

def convolutional_parameter_search(train_loader: DataLoader,
                     val_loader: DataLoader,
                     model_fn:Callable[[], nn.Module]) -> float:
    """
    Parameter search for our linear model using SGD.

    Args:
    train_loader: the train dataloader.
    val_loader: the validation dataloader.
    model_fn: a function that, when called, returns a torch.nn.Module.

    Returns:
    The learning rate with the least validation loss.
    """
    num_iter = 10
    num_choices = 30
    best_loss = torch.tensor(np.inf)
    best_lr = 0.0
    best_momentum = 0

    best_M = 0
    best_k = 0
    best_N = 0

    lrs = torch.linspace(10 ** (-6), 10 ** (-1), num_choices)
    momentums = torch.linspace(0.5, 0.99, num_choices)

    hidden_layers = torch.linspace(100, 999, num_choices).int()
    kernel_sizes = [4, 5, 6, 7]
    max_pools = torch.linspace(2, 15, 15).int()


    for _ in range(num_iter):
        lr = random.choice(lrs)
        momentum = random.choice(momentums)

        M = int(random.choice(hidden_layers))
        k = int(random.choice(kernel_sizes))
        N = int(random.choice(max_pools))

        print(f"trying learning rate {lr} and momentum {momentum} and M {M} and k {k} and N {N}")
        model = model_fn(M, k, N)
        optim = SGD(model.parameters(), lr=lr, momentum=momentum)
        train_loss, train_acc, val_loss, val_acc = train(
            model,
            optim,
            train_loader,
            val_loader,
            epochs=20
            )

        if min(val_loss) < best_loss:
            best_loss = min(val_loss)
            best_lr = lr
            best_momentum = momentum

            best_M = M
            best_k = k
            best_N = N

    return best_lr, best_momentum, best_M, best_k, best_N

#find best learning rate and hidden layer size
best_lr, best_momentum, best_M, best_k, best_N = convolutional_parameter_search(train_loader, val_loader, convolutional_model)

# best_lr, best_momentum, best_M, best_k, best_N = 0.013793965801596642,0.9224138259887695, 534, 4, 9
print(f"best learning rate: {best_lr} best momentum: {best_momentum} best M: {best_M} best k: {best_k} best N: {best_N}")
model = convolutional_model(best_M, best_k, best_N)
optimizer = SGD(model.parameters(), lr=best_lr, momentum=best_momentum)

train_loss, train_accuracy, val_loss, val_accuracy = train(
    model, optimizer, train_loader, val_loader, 20)

epochs = range(1, 21)
plt.plot(epochs, train_accuracy, label="Train Accuracy")
plt.plot(epochs, val_accuracy, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Logistic Regression Accuracy for CIFAR-10 vs Epoch")
plt.show()

test_loss, test_acc = evaluate(model, test_loader)
print(f"Test Accuracy: {test_acc}")

#Make Plots of 3 best hyperparemeter sets

#graph 1 (best)
best_lr, best_momentum, best_M, best_k, best_N = 0.013793965801596642,0.9224138259887695, 534, 4, 9
model = convolutional_model(best_M, best_k, best_N)
optimizer = SGD(model.parameters(), lr=best_lr, momentum=best_momentum)
train_loss_1, train_accuracy_1, val_loss_1, val_accuracy_1 = train(
    model, optimizer, train_loader, val_loader, 20)

#graph 2
best_lr, best_momentum, best_M, best_k, best_N = 0.024138689041137695, 0.6689655184745789, 658, 6, 9
model = convolutional_model(best_M, best_k, best_N)
optimizer = SGD(model.parameters(), lr=best_lr, momentum=best_momentum)
train_loss_2, train_accuracy_2, val_loss_2, val_accuracy_2 = train(
    model, optimizer, train_loader, val_loader, 20)

#graph 3
best_lr, best_momentum, best_M, best_k, best_N = 0.027586931362748146, 0.7365517020225525, 162, 7, 5
model = convolutional_model(best_M, best_k, best_N)
optimizer = SGD(model.parameters(), lr=best_lr, momentum=best_momentum)
train_loss_3, train_accuracy_3, val_loss_3, val_accuracy_3 = train(
    model, optimizer, train_loader, val_loader, 20)

plt.figure(figsize=(12, 6))
plt.plot(range(1, 21), train_accuracy_1, label="train accuracy: best learning rate: 0.013793965801596642 best momentum: 0.9224138259887695 best M: 534 best k: 4 best N: 9")
plt.plot(range(1, 21), val_accuracy_1, label="val accuracy: best learning rate: 0.013793965801596642 best momentum: 0.9224138259887695 best M: 534 best k: 4 best N: 9", linestyle='dashed')

plt.plot(range(1, 21), train_accuracy_2, label="train accuracy: best learning rate: 0.024138689041137695 best momentum: 0.6689655184745789 best M: 658 best k: 6 best N: 9")
plt.plot(range(1, 21), val_accuracy_2, label="val accuracy: best learning rate: 0.024138689041137695 best momentum: 0.6689655184745789 best M: 658 best k: 6 best N: 9", linestyle='dashed')

plt.plot(range(1, 21), train_accuracy_3, label="train accuracy: best learning rate: 0.027586931362748146 best momentum: 0.7365517020225525 best M: 162 best k: 7 best N: 5")
plt.plot(range(1, 21), val_accuracy_3, label="val accuracy: best learning rate: 0.027586931362748146 best momentum: 0.7365517020225525 best M: 162 best k: 7 best N: 5", linestyle='dashed')

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid('on')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
plt.axhline(0.65, linestyle='--')
plt.show()