import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import os

from utils.data_loader import load_mnist, visualize_means, plot_log_likelihood, evaluate_clustering
from em_algorithm import GaussianMixtureEM

def main(args):
    print("=== EM Algorithm for MNIST ===")
    print(f"Number of components: {args.n_components}")
    print(f"PCA components: {args.pca_components}")
    print(f"Max iterations: {args.max_iter}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MNIST data
    print("\nLoading MNIST dataset...")
    X_train, y_train = load_mnist(
        root=args.data_dir, 
        download=True, 
        train=True, 
        pca_components=args.pca_components
    )
    
    X_test, y_test = load_mnist(
        root=args.data_dir, 
        download=True, 
        train=False, 
        pca_components=args.pca_components
    )
    
    print(f"Train data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test data: {X_test.shape}, Labels: {y_test.shape}")
    
    # Initialize and fit the model
    print("\nInitializing EM algorithm...")
    n_features = X_train.shape[1]
    model = GaussianMixtureEM(
        n_components=args.n_components, 
        n_features=n_features,
        tol=args.tol,
        max_iter=args.max_iter,
        random_state=args.random_state
    )
    
    print("\nFitting the model (this may take a while)...")
    start_time = time.time()
    model.fit(X_train)
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")
    
    # Evaluate on training set
    print("\nEvaluating on training set...")
    train_pred = model.predict(X_train)
    train_accuracy, train_confusion = evaluate_clustering(y_train, train_pred, n_clusters=args.n_components)
    print(f"Training accuracy: {train_accuracy:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_pred = model.predict(X_test)
    test_accuracy, test_confusion = evaluate_clustering(y_test, test_pred, n_clusters=args.n_components)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Plot component means
    print("\nVisualizing component means...")
    if args.pca_components is None:
        # We can visualize means as images if we're in original space
        fig = visualize_means(model.means, image_shape=(28, 28), title="Gaussian Component Means")
        fig.savefig(os.path.join(args.output_dir, "component_means.png"))
    
    # Plot log-likelihood history
    print("\nPlotting log-likelihood convergence...")
    fig = plot_log_likelihood(model.log_likelihood_history, title="EM Algorithm Convergence")
    fig.savefig(os.path.join(args.output_dir, "log_likelihood.png"))
    
    # Plot confusion matrix for test set
    print("\nPlotting confusion matrix...")
    plt.figure(figsize=(10, 8))
    plt.imshow(test_confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Test Set)")
    plt.colorbar()
    tick_marks = np.arange(args.n_components)
    plt.xticks(tick_marks, range(args.n_components), rotation=45)
    plt.yticks(tick_marks, range(args.n_components))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"))
    
    print(f"\nResults saved to {args.output_dir}")
    
    # Show all plots
    if args.show_plots:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Expectation-Maximization for MNIST")
    
    parser.add_argument("--n_components", type=int, default=10,
                        help="Number of Gaussian components")
    parser.add_argument("--pca_components", type=int, default=None,
                        help="Number of PCA components (None for no PCA)")
    parser.add_argument("--max_iter", type=int, default=50,
                        help="Maximum number of EM iterations")
    parser.add_argument("--tol", type=float, default=1e-3,
                        help="Convergence tolerance")
    parser.add_argument("--random_state", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Directory to store MNIST data")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--show_plots", action="store_true",
                        help="Show plots after saving")
    
    args = parser.parse_args()
    main(args) 