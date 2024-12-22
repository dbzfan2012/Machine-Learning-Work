"""
We will be building a personalized movie recommendation system using SVD and Python with Python packages such as NumPy and PyTorch.

We will use the 100K MovieLens dataset available at https://grouplens.org/datasets/movielens/100k/ to estimate unknown user ratings given their previous ratings. Run the code block below to download the dataset.
"""
#Use this to download the dataset
# # @title loading dataset
# !rm -rf ml-100k*
# !wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
# !unzip ml-100k.zip
# !mv ml-100k/u.data .

import csv
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import torch

"""Now, let's load the 100K MovieLens data."""

data = []
with open('u.data') as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\t')
    for row in spamreader:
        data.append([int(row[0])-1, int(row[1])-1, int(row[2])])
data = np.array(data)

num_observations = len(data)  # num_observations = 100,000
num_users = max(data[:,0])+1  # num_users = 943, indexed 0,...,942
num_items = max(data[:,1])+1  # num_items = 1682 indexed 0,...,1681

np.random.seed(1)
num_train = int(0.8*num_observations)
perm = np.random.permutation(data.shape[0])
train = data[perm[0:num_train],:]
test = data[perm[num_train::],:]

print(f"Successfully loaded 100K MovieLens dataset with",
      f"{len(train)} training samples and {len(test)} test samples")

"""Here, we are computing estimates via a matrix R comprised of the averages of each movie for every user
For every i,j in R, where the columns represents the user and the rows the movie, each user will give the movie
a review equal to the average across the entire training data for that movie"""
# Compute estimate
total_movie_ratings = np.zeros(1682) #summed rating of each movie
total_movie_watchers = np.zeros(1682) #summed number of watchers for each movie
total_watchers = 0

watcher_ = -1
for watcher, movie, score in train:
    total_movie_ratings[movie] += score
    total_movie_watchers[movie] += 1
    total_watchers += 1

average_scores = np.zeros(num_items)
for i in range(num_items):
    average_score = np.round(total_movie_ratings[i] / total_movie_watchers[i]) if total_movie_watchers[i] != 0 else 0
    average_scores[i] = average_score

R_hat = np.tile(average_scores.reshape(-1, 1), total_watchers)
print(R_hat)

# Evaluate test error
errors = []
for watcher, movie, score in test:
    predicted_score = R_hat[movie, watcher]
    errors.append((predicted_score - score) ** 2)
test_error = np.mean(errors)
print(test_error)

"""### Part (b)
Here we allocate a matrix R_twiddle and set its entries equal to the known values in the training set, and 0 otherwise.
Let R be the best rank-d approximation (in terms of squared error) approximation to R_twiddle. This is equivalent to computing the singular value decomposition (SVD) and using the top d singular values. This learns a lower-dimensional vector representation for users and movies, assuming that each user would give a rating of 0 to any movie they have not reviewed.
"""

# Create the matrix R_twiddle
r_twiddle = np.zeros((1682, 943))
for watcher, movie, score in train:
    r_twiddle[movie, watcher] = score
print(r_twiddle)


def construct_estimator(d, r_twiddle):
  U, sigma, V_T = svds(r_twiddle, k=d)
  return U @ np.diag(sigma) @ V_T


def get_error(d, r_twiddle, dataset):
  estimator = construct_estimator(d, r_twiddle)
  errors = []
  for watcher, movie, score in dataset:
    predicted_score = estimator[movie, watcher]
    errors.append((predicted_score - score) ** 2)
  return np.mean(errors)


# Evaluate train and test error for: d = 1, 2, 5, 10, 20, 50.
d_values = [1, 2, 5, 10, 20, 50]
train_errors = []
test_errors = []
for d in d_values:
    train_error = get_error(d, r_twiddle, train)
    test_error = get_error(d, r_twiddle, test)

    train_errors.append(train_error)
    test_errors.append(test_error)

print(train_errors)
print(test_errors)


# Plot both train and test error as a function of d on the same plot.
plt.figure(figsize=(8, 6))
plt.plot(d_values, train_errors, label='Train MSE', marker='o')
plt.plot(d_values, test_errors, label='Test MSE', marker='o')
plt.xlabel("d")
plt.ylabel("Error")
plt.title("MSE vs d")
plt.legend()
plt.show()

"""### Part (c)
Replacing all missing values by a constant may impose strong and potentially incorrect assumptions on the unobserved entries of R. Instead, minimize the mean squared error (MSE) only on rated movies. Define a loss function:
$$
\mathcal{L} \left( \{u_i\}_{i=1}^m, \{v_j\}_{j=1}^n \right) :=
\sum_{(i, j, R_{i, j}) \in {\rm train}} (\langle u_i,v_j\rangle - R_{i,j})^2 +
\lambda \sum_{i=1}^m \|u_i\|_2^2 +
\lambda \sum_{j=1}^n \|v_j\|_2^2
$$
where $\lambda > 0$ is the regularization coefficient. We will implement algorithms to learn vector representations by minimizing the above loss.
"""

r_i = {i : [] for i in range(1682)}
r_j = {i : [] for i in range(943)}
for watcher, movie, score in train:
    r_i[movie].append(watcher)
    r_j[watcher].append(movie)

def closed_form_u(V, U, l, r_twiddle):
    regularization = l * np.eye(U.shape[1])
    for i in range(U.shape[0]):
        watched = r_i[i]
        V_watched = V[watched]
        R_watched = r_twiddle[i, watched]
        VtV = V_watched.T @ V_watched
        U[i] = np.linalg.solve(VtV + regularization, V_watched.T @ R_watched)
    return U

def closed_form_v(V, U, l, r_twiddle):
    regularization = l * np.eye(U.shape[1])
    for j in range(V.shape[0]):
        movies = r_j[j]
        U_movies = U[movies]
        R_movies = r_twiddle[movies, j]
        UtU = U_movies.T @ U_movies
        V[j] = np.linalg.solve(UtU + regularization, U_movies.T @ R_movies)
    return V

def construct_alternating_estimator(d, r_twiddle, l=10.0, delta=1e-1, sigma=0.1):
    m, n = r_twiddle.shape
    U = np.random.randn(m, d)
    V = np.random.randn(n, d)

    is_converged = False
    epoch = 1
    while not is_converged:
        U_prev, V_prev = U.copy(), V.copy()

        U = closed_form_u(V, U, l, r_twiddle)
        V = closed_form_v(V, U, l, r_twiddle)

        is_converged = check_convergence(U, U_prev, V, V_prev, delta)
        epoch += 1
    print(f"Converged at epoch {epoch}")
    return U, V

def check_convergence(U, U_prev, V, V_prev, delta):
    return np.linalg.norm(U - U_prev) < delta and np.linalg.norm(V - V_prev) < delta

def get_error_c(r_hat, dataset):
  errors = []
  for watcher, movie, score in dataset:
    predicted_score = r_hat[movie, watcher]
    errors.append((predicted_score - score) ** 2)
  return np.mean(errors)

# Evaluate train and test error for: d = 1, 2, 5, 10, 20, 50.
import random

d_values = [1, 2, 5, 10, 20, 50]

# Hyperparameters
l = 15
sigma = 0.5

train_errors = []
test_errors = []
print(f"Using sigma {sigma} and lambda {l}")
for d in d_values:
    U, V = construct_alternating_estimator(d, r_twiddle, l=l, sigma=sigma)
    r_hat = U @ V.T
    train_error = get_error_c(r_hat, train)
    test_error = get_error_c(r_hat, test)

    train_errors.append(train_error)
    test_errors.append(test_error)

print(train_errors)
print(test_errors)

# Plot both train and test error as a function of d on the same plot.
plt.figure(figsize=(8, 6))
plt.plot(d_values, train_errors, label='Train MSE', marker='o')
plt.plot(d_values, test_errors, label='Test MSE', marker='o')
plt.xlabel("d")
plt.ylabel("Error")
plt.title("MSE vs d")
plt.legend()
plt.show()