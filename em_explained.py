"""
Explanation of the Expectation-Maximization (EM) Algorithm

This script provides a simplified explanation of how the EM algorithm works
for Gaussian Mixture Models, with detailed comments explaining each step.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import load_mnist
from em_algorithm import GaussianMixtureEM

def explain_em_with_mnist():
    """
    Demonstrate the EM algorithm step-by-step with explanations
    using a small subset of the MNIST dataset
    """
    print("=" * 80)
    print("UNDERSTANDING EXPECTATION-MAXIMIZATION WITH MNIST")
    print("=" * 80)
    
    # Load a small subset of MNIST for demonstration
    X, y = load_mnist(download=True, train=True)
    
    # Use only 500 samples and reduce dimensionality for visualization
    n_samples = 500
    pca_components = 2  # Use 2 for visualization
    
    # Apply PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=pca_components, random_state=42)
    X_reduced = pca.fit_transform(X[:n_samples])
    y_subset = y[:n_samples]
    
    print(f"\nUsing {n_samples} samples with {pca_components} PCA components")
    print(f"Data shape: {X_reduced.shape}")
    
    # Visualize the data
    plt.figure(figsize=(10, 8))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_subset, cmap='tab10', alpha=0.6)
    plt.title("MNIST dataset projected to 2D using PCA")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.colorbar(label="Digit class")
    plt.savefig("pca_mnist.png")
    
    # ========================================================================
    # STEP 1: EXPLANATION OF THE EM ALGORITHM
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: HOW EM WORKS - THEORY")
    print("=" * 80)
    
    print("""
The Expectation-Maximization (EM) algorithm is an iterative method for finding
maximum likelihood estimates of parameters in statistical models with latent variables.

For Gaussian Mixture Models (GMMs), the EM algorithm consists of two steps:

1. E-step (Expectation):
   - Compute the "responsibility" of each component for each data point
   - These responsibilities are the posterior probabilities of component membership
   - For a given data point x, the responsibility of component k is:
     γ(z_k) = p(z_k|x) = [π_k * N(x|μ_k,Σ_k)] / [Σⱼ π_j * N(x|μ_j,Σ_j)]
     where:
     * π_k is the mixture weight for component k
     * N(x|μ_k,Σ_k) is the probability density of x under a Gaussian with mean μ_k and covariance Σ_k

2. M-step (Maximization):
   - Update the model parameters to maximize the expected log-likelihood
   - The updated parameters based on the responsibilities are:
     * New means: μ_k = Σᵢ γ(z_kⁱ) * xⁱ / Σᵢ γ(z_kⁱ)
     * New covariances: Σ_k = Σᵢ γ(z_kⁱ) * (xⁱ - μ_k)(xⁱ - μ_k)ᵀ / Σᵢ γ(z_kⁱ)
     * New weights: π_k = Σᵢ γ(z_kⁱ) / N

The algorithm iterates between these steps until convergence, measured by the change in log-likelihood.
    """)
    
    # ========================================================================
    # STEP 2: DEMONSTRATION OF EM INITIALIZATION
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: INITIALIZATION")
    print("=" * 80)
    
    # For simplicity, we'll use 3 components (even though MNIST has 10 classes)
    n_components = 3
    
    print(f"\nInitializing a GMM with {n_components} components")
    print("In a real-world scenario, we'd typically use 10 components for MNIST (one per digit).")
    print("For visualization purposes, we're using fewer components in 2D space.")
    
    # Initialize model
    model = GaussianMixtureEM(
        n_components=n_components, 
        n_features=X_reduced.shape[1],
        max_iter=1,  # We'll manually perform iterations for explanation
        random_state=42
    )
    
    # Initialize random parameters
    np.random.seed(42)
    init_means = torch.tensor(
        np.random.randn(n_components, X_reduced.shape[1]) * 2,
        dtype=torch.float32
    )
    
    # Initialize with identity covariances
    init_covs = torch.stack([
        torch.eye(X_reduced.shape[1]) for _ in range(n_components)
    ])
    
    # Equal weights initially
    init_weights = torch.ones(n_components) / n_components
    
    model.means = init_means.to(model.device)
    model.covs = init_covs.to(model.device)
    model.weights = init_weights.to(model.device)
    
    print("\nInitial Parameters:")
    print(f"- Means:\n{model.means.cpu().numpy()}")
    print(f"- Weights: {model.weights.cpu().numpy()}")
    
    # Visualize initial state
    plt.figure(figsize=(10, 8))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_subset, cmap='tab10', alpha=0.3)
    
    # Plot initial Gaussians
    for k in range(n_components):
        mean = model.means[k].cpu().numpy()
        plt.scatter(mean[0], mean[1], s=200, marker='*', color=f'C{k}', edgecolor='black')
        
        # Draw ellipses representing covariances
        from matplotlib.patches import Ellipse
        cov = model.covs[k].cpu().numpy()
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * np.sqrt(5.991 * eigvals)  # 95% confidence interval
        
        ellipse = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            fill=False, edgecolor=f'C{k}', linewidth=2
        )
        plt.gca().add_patch(ellipse)
    
    plt.title("Initial Gaussian Components")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.savefig("em_init.png")
    
    # ========================================================================
    # STEP 3: PERFORM E-STEP (EXPECTATION)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: E-STEP (EXPECTATION)")
    print("=" * 80)
    
    print("""
In the E-step, we compute the 'responsibility' that each Gaussian component takes
for explaining each data point. The responsibility is the posterior probability
that the component generated the data point.
    """)
    
    # Convert data to tensor
    X_tensor = torch.tensor(X_reduced, dtype=torch.float32).to(model.device)
    
    # Compute responsibilities (posterior probabilities)
    responsibilities, log_likelihood = model._e_step(X_tensor)
    responsibilities_np = responsibilities.cpu().numpy()
    
    print(f"\nResponsibilities shape: {responsibilities_np.shape}")
    print(f"This means each of the {n_samples} data points has a probability distribution")
    print(f"over the {n_components} Gaussian components.")
    
    # Display the first few responsibilities
    print("\nFirst 5 data points with their responsibilities:")
    for i in range(5):
        print(f"Data point {i} (digit {y_subset[i]}):")
        for k in range(n_components):
            print(f"  - Component {k}: {responsibilities_np[i, k]:.4f}")
    
    print(f"\nLog-likelihood after E-step: {log_likelihood.item():.4f}")
    
    # Visualize the responsibilities
    plt.figure(figsize=(12, 5))
    
    # Get most responsible component for each point
    resp_max = np.argmax(responsibilities_np, axis=1)
    
    # Plot data colored by most responsible component
    plt.subplot(1, 2, 1)
    for k in range(n_components):
        idx = resp_max == k
        plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], c=f'C{k}', label=f'Component {k}')
    
    # Plot means
    for k in range(n_components):
        mean = model.means[k].cpu().numpy()
        plt.scatter(mean[0], mean[1], s=200, marker='*', color=f'C{k}', edgecolor='black')
    
    plt.title("Data Points Colored by Most Responsible Component")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    
    # Plot responsibilities as a heatmap for a few points
    plt.subplot(1, 2, 2)
    plt.imshow(responsibilities_np[:20].T, aspect='auto', cmap='Blues')
    plt.colorbar(label="Responsibility")
    plt.xlabel("Data Point")
    plt.ylabel("Component")
    plt.yticks(range(n_components))
    plt.title("Responsibilities for First 20 Points")
    
    plt.tight_layout()
    plt.savefig("e_step.png")
    
    # ========================================================================
    # STEP 4: PERFORM M-STEP (MAXIMIZATION)
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: M-STEP (MAXIMIZATION)")
    print("=" * 80)
    
    print("""
In the M-step, we update the parameters of each Gaussian component to maximize
the expected log-likelihood, based on the responsibilities computed in the E-step.
    """)
    
    # Save old parameters for comparison
    old_means = model.means.clone()
    old_covs = model.covs.clone()
    old_weights = model.weights.clone()
    
    # Perform M-step
    new_means, new_covs, new_weights = model._m_step(X_tensor, responsibilities)
    
    print("\nParameter updates after M-step:")
    print("Means:")
    for k in range(n_components):
        print(f"  Component {k}:")
        print(f"    - Old: {old_means[k].cpu().numpy()}")
        print(f"    - New: {new_means[k].cpu().numpy()}")
        
    print("\nWeights:")
    print(f"  - Old: {old_weights.cpu().numpy()}")
    print(f"  - New: {new_weights.cpu().numpy()}")
    
    # Update model parameters
    model.means = new_means
    model.covs = new_covs
    model.weights = new_weights
    
    # Visualize the updated model
    plt.figure(figsize=(10, 8))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_subset, cmap='tab10', alpha=0.3)
    
    # Plot old and new means
    for k in range(n_components):
        old_mean = old_means[k].cpu().numpy()
        new_mean = model.means[k].cpu().numpy()
        
        plt.scatter(old_mean[0], old_mean[1], s=150, marker='o', color=f'C{k}', alpha=0.5, label=f'Old Mean {k}' if k==0 else "")
        plt.scatter(new_mean[0], new_mean[1], s=200, marker='*', color=f'C{k}', edgecolor='black', label=f'New Mean {k}' if k==0 else "")
        
        # Draw arrow from old to new mean
        plt.arrow(old_mean[0], old_mean[1], new_mean[0]-old_mean[0], new_mean[1]-old_mean[1], 
                 head_width=0.1, head_length=0.2, fc=f'C{k}', ec=f'C{k}', length_includes_head=True)
        
        # Draw ellipses for new covariances
        from matplotlib.patches import Ellipse
        cov = model.covs[k].cpu().numpy()
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * np.sqrt(5.991 * eigvals)  # 95% confidence interval
        
        ellipse = Ellipse(
            xy=new_mean, width=width, height=height, angle=angle,
            fill=False, edgecolor=f'C{k}', linewidth=2
        )
        plt.gca().add_patch(ellipse)
    
    plt.title("Updated Gaussian Components after M-step")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    plt.savefig("m_step.png")
    
    # ========================================================================
    # STEP 5: COMPLETE EM ALGORITHM
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: COMPLETE EM ALGORITHM")
    print("=" * 80)
    
    print("""
The EM algorithm iterates between the E-step and M-step until convergence.
We now run the full algorithm for multiple iterations to see how the model converges.
    """)
    
    # Create a new model and run for more iterations
    model_full = GaussianMixtureEM(
        n_components=n_components, 
        n_features=X_reduced.shape[1],
        max_iter=20,
        tol=1e-4,
        random_state=42
    )
    
    # Fit the model
    print("\nRunning full EM algorithm...")
    model_full.fit(X_reduced)
    
    print(f"\nConverged after {len(model_full.log_likelihood_history)} iterations")
    print(f"Final log-likelihood: {model_full.log_likelihood_history[-1]:.4f}")
    
    # Plot log-likelihood convergence
    plt.figure(figsize=(10, 5))
    plt.plot(model_full.log_likelihood_history)
    plt.title("Log-Likelihood Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    plt.savefig("convergence.png")
    
    # Visualize final model
    plt.figure(figsize=(10, 8))
    
    # Predict cluster assignments
    predictions = model_full.predict(X_reduced)
    
    # Plot data colored by predicted cluster
    for k in range(n_components):
        idx = predictions == k
        plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], c=f'C{k}', label=f'Cluster {k}')
    
    # Plot final means and covariances
    for k in range(n_components):
        mean = model_full.means[k].cpu().numpy()
        plt.scatter(mean[0], mean[1], s=200, marker='*', color=f'C{k}', edgecolor='black')
        
        # Draw ellipses for covariances
        from matplotlib.patches import Ellipse
        cov = model_full.covs[k].cpu().numpy()
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 2 * np.sqrt(5.991 * eigvals)  # 95% confidence interval
        
        ellipse = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            fill=False, edgecolor=f'C{k}', linewidth=2
        )
        plt.gca().add_patch(ellipse)
    
    plt.title("Final Clustering Result")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.legend()
    plt.savefig("final_clustering.png")
    
    # ========================================================================
    # STEP 6: EVALUATE AND INTERPRET RESULTS
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: EVALUATING RESULTS")
    print("=" * 80)
    
    # Evaluate clustering against true labels
    from utils.data_loader import evaluate_clustering
    accuracy, confusion = evaluate_clustering(y_subset, predictions, n_clusters=n_components)
    
    print(f"\nClustering Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(confusion)
    
    print("""
INTERPRETATION:

1. The EM algorithm identified clusters in the data, but since we used fewer components (3)
   than the actual number of classes (10), each cluster represents multiple digit classes.

2. The confusion matrix shows which digits were grouped together. Some digits that are
   visually similar (like 3 and 5, or 7 and 9) may end up in the same cluster.

3. In the full MNIST application, we would use 10 components (one per digit) and work
   in higher dimensions to achieve better separation between the digits.
   
4. The model converges as the log-likelihood stabilizes, indicating that the EM algorithm
   has found a locally optimal solution.
   
PRACTICAL CONSIDERATIONS:

1. Initialization matters - random initialization can lead to different local optima.
   K-means initialization is often used to get better starting points.

2. The number of components needs to be chosen carefully. Using too few components
   leads to underfitting, while too many can lead to overfitting.

3. High-dimensional data (like the full 784 dimensions of MNIST) can suffer from the
   "curse of dimensionality." Dimensionality reduction (like PCA) can help.

4. For large datasets, the EM algorithm can be computationally expensive. Mini-batch
   versions can be used for better scalability.
    """)
    
    print("\nAll visualizations have been saved as PNG files in the current directory.")
    print("This concludes the explanation of the EM algorithm with MNIST data.")

if __name__ == "__main__":
    explain_em_with_mnist() 