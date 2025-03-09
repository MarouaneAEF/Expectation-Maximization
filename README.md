# Expectation-Maximization (EM) Algorithm for MNIST

This project provides an efficient PyTorch implementation of the Expectation-Maximization (EM) algorithm applied to the MNIST handwritten digits dataset. The implementation uses Gaussian Mixture Models (GMMs) to model the data distribution.

## Overview

The Expectation-Maximization algorithm is an iterative method for finding maximum likelihood estimates of parameters in statistical models with latent variables. In this implementation, we use EM to fit a Gaussian Mixture Model to the MNIST dataset, which consists of 28x28 grayscale images of handwritten digits (0-9).

## Features

- Complete implementation of the EM algorithm for Gaussian Mixture Models
- Integration with MNIST dataset loading via PyTorch
- Support for dimensionality reduction using PCA
- Comprehensive evaluation and visualization tools
- GPU acceleration when available

## Requirements

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

To run the EM algorithm on MNIST with default settings:

```bash
python main.py
```

This will:
1. Download the MNIST dataset (if not already present)
2. Train a GMM with 10 components using the EM algorithm
3. Evaluate the clustering performance 
4. Generate visualizations in the `./results` directory

### Advanced Usage

Command-line arguments allow you to customize the behavior:

```bash
python main.py --n_components 15 --pca_components 50 --max_iter 100 --output_dir ./custom_results
```

#### Available Arguments

- `--n_components`: Number of Gaussian components (default: 10)
- `--pca_components`: Number of PCA components for dimensionality reduction (default: None, uses full 784 dimensions)
- `--max_iter`: Maximum number of EM iterations (default: 50)
- `--tol`: Convergence tolerance (default: 1e-3)
- `--random_state`: Random seed for reproducibility (default: 42)
- `--data_dir`: Directory to store MNIST data (default: ./data)
- `--output_dir`: Directory to save results (default: ./results)
- `--show_plots`: Display plots after saving (default: False)

## Implementation Details

The EM algorithm implementation follows these steps:

1. **Initialization**: Initialize means, covariances, and weights for the Gaussian components
2. **E-step**: Compute the "responsibilities" of each component for each data point
3. **M-step**: Update the parameters (means, covariances, weights) based on the responsibilities
4. **Repeat** until convergence or max iterations reached

## PCA Dimensionality Reduction

For faster computation, you can use PCA to reduce the dimensionality of the data. For example:

```bash
python main.py --pca_components 50
```

This reduces the data from 784 dimensions (28x28 pixels) to the specified number of principal components, greatly accelerating training while preserving most of the variance.

## Performance

Using the default settings on a modern CPU, training should complete in a few minutes. Using a GPU will significantly accelerate training. The accuracy of the clustering depends on several factors, including the number of components and dimensionality reduction settings.

## Understanding the Results

After running the algorithm, check the `results` directory for:

1. `component_means.png`: Visualization of the learned Gaussian means (interpretable as digit prototypes)
2. `log_likelihood.png`: Convergence plot of the log-likelihood
3. `confusion_matrix.png`: Confusion matrix showing how well the clusters align with the true digit classes

## License

This project is open source and available under the MIT License. 