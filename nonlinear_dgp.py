import numpy as np


def univariate_gaussian_process(x, a=1, l=1):
    """
    Evaluate the univariate Gaussian process at given input points.

    Parameters:
    - x (array-like): Input points where the Gaussian process will be evaluated.
    - a (float, optional): Amplitude parameter. Determines the height of the Gaussian function.
                          Default is 1.
    - l (float, optional): Lengthscale parameter. Controls the smoothness of the Gaussian function.
                          Default is 1.

    Returns:
    - y (numpy.ndarray): Array containing the values of the Gaussian process evaluated at input points.
    """
    # In a Gaussian Process, the covariance matrix is defined using the RBF kernel.
    # https://d2l.ai/chapter_gaussian-processes/gp-intro.html <- check this for notation
    x = x.reshape(-1, 1)
    diff_matrix = x - x.T
    cov = a**2 * np.exp(-(diff_matrix**2) / (2 * l**2))
    mean = np.zeros_like(x.flatten())
    return np.random.multivariate_normal(mean, cov)


def generate_gp_cam(n_features, n_samples=1000, sparseness=0, num_confounders=0):
    confounders = [
        np.random.randn(n_samples) * np.sqrt(np.random.uniform(1, 3))
        for _ in range(num_confounders)
    ]
    features_to_confound = []
    while len(features_to_confound) < len(confounders):
        # must at least be 2
        features = set(np.random.choice(range(n_features), 2, replace=False))
        features_to_confound.append(features)

    # Gaussian Process, Causal Additive Models
    # Sparseness will skip some edges with some probability as specified.
    features = [np.random.randn(n_samples) * np.sqrt(np.random.uniform(1, 3))]
    for conf_idx, conf_feat in enumerate(features_to_confound):
        if 0 in conf_feat:
            features[0] += confounders[conf_idx]

    while len(features) < n_features:
        new_feature = np.sum(
            [
                univariate_gaussian_process(x)
                * np.random.binomial(1, 1 - sparseness, 1)
                for x in features
            ],
            axis=0,
        ) + np.random.randn(n_samples) * np.sqrt(np.random.uniform(1, 3))

        # add confounders
        for conf_idx, conf_feat in enumerate(features_to_confound):
            if len(features) in conf_feat:
                new_feature += confounders[conf_idx]

        features.append(new_feature)

    return np.vstack(features).T



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generic_nn_function(X):
    n, m = X.shape

    # Randomly initialize the weights and biases for the neural network
    # np.random.seed(0)  # For reproducibility, you can remove this in practice

    n_hidden_units = [m, m // 2 + 1]

    W1 = np.random.randn(m, n_hidden_units[0])
    b1 = np.random.randn(n_hidden_units[0])
    W2 = np.random.randn(n_hidden_units[0], n_hidden_units[1])
    b2 = np.random.randn(n_hidden_units[1])
    W3 = np.random.randn(n_hidden_units[1], 1)
    b3 = np.random.randn(1)

    # Forward pass through the neural network with batch normalization
    hidden_layer1 = np.dot(X, W1) + b1
    hidden_layer1 = sigmoid(hidden_layer1)
    hidden_layer2 = np.dot(hidden_layer1, W2) + b2
    hidden_layer2 = sigmoid(hidden_layer2)
    output = np.dot(hidden_layer2, W3) + b3

    return output.flatten()

def generate_data_nn_anm(n_features, n_samples=100, sparseness=0.3):
    features = [np.random.randn(n_samples)* np.sqrt(np.random.uniform(1,3))]
    while len(features) < n_features:
        new_feature = [f for f in features if np.random.binomial(1,1-sparseness, 1).item()]
        new_feature = generic_nn_function(np.vstack(features).T) + np.random.randn(
            n_samples
        ) * np.sqrt(np.random.uniform(1, 3))
        features.append(new_feature)
    return np.vstack(features).T