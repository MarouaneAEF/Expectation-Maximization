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

## Visualizations and Understanding the Results

The implementation generates several visualizations to help understand the EM algorithm and its results:

### Main Script Visualizations (`main.py`)

When you run the main script, the following visualizations are generated in the output directory:

1. **Component Means (`component_means.png`)**:
   - Visual representation of the learned Gaussian component means
   - Each component mean is displayed as a 28x28 image
   - These can be interpreted as the "prototype" digits discovered by the algorithm
   - When using the full dimensionality, these should resemble actual handwritten digits

   ![Component Means](https://i.imgur.com/example1.png)

2. **Log-Likelihood Convergence (`log_likelihood.png`)**:
   - Plot showing the log-likelihood value at each iteration
   - Helps monitor the convergence of the EM algorithm
   - A plateau indicates that the algorithm has converged to a (local) optimum

   ![Log-Likelihood](https://i.imgur.com/example2.png)

3. **Confusion Matrix (`confusion_matrix.png`)**:
   - Shows the relationship between the discovered clusters and the true digit classes
   - Bright diagonal elements indicate good correspondence between clusters and classes
   - Off-diagonal elements indicate digits that were clustered together

   ![Confusion Matrix](https://i.imgur.com/example3.png)

### Educational Visualizations (`em_explained.py`)

For a deeper understanding of the EM algorithm, the `em_explained.py` script generates step-by-step visualizations:

1. **PCA Projection (`pca_mnist.png`)**:
   - MNIST digits projected to 2D using PCA
   - Different colors represent different true digit classes
   - Provides intuition about the structure of the data

2. **Initial Components (`em_init.png`)**:
   - Initial random Gaussian components before EM iterations
   - Shows starting point of the algorithm

3. **E-Step Visualization (`e_step.png`)**:
   - Illustrates the computation of responsibilities
   - Shows which component is responsible for which data points
   - Includes a heatmap of responsibility values

4. **M-Step Visualization (`m_step.png`)**:
   - Shows how means and covariances are updated
   - Arrows indicate the movement of means from old to new positions

5. **Convergence Plot (`convergence.png`)**:
   - Shows the convergence of log-likelihood over multiple iterations

6. **Final Clustering (`final_clustering.png`)**:
   - Final cluster assignments with updated Gaussian components
   - Visual representation of the clustering result

These visualizations are powerful tools for understanding both the EM algorithm itself and the structure of the MNIST dataset.

## Educational Value

This implementation serves both as a practical tool for clustering MNIST digits and as an educational resource for understanding the EM algorithm. The `em_explained.py` script provides a detailed walkthrough of the algorithm with intuitive visualizations.

## License

This project is open source and available under the MIT License. 