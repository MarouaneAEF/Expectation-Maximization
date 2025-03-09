import torch
import numpy as np
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

def load_mnist(root='./data', download=True, train=True, pca_components=None, batch_size=None):
    """
    Load MNIST dataset and optionally apply dimensionality reduction
    
    Args:
        root: Root directory for data
        download: Whether to download data
        train: Whether to load train or test set
        pca_components: Number of components for PCA dimensionality reduction
        batch_size: Batch size for DataLoader (if None, returns all data as tensors)
        
    Returns:
        If batch_size is None: X (features), y (labels)
        If batch_size is set: DataLoader instance
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    # Load dataset
    dataset = datasets.MNIST(root=root, train=train, download=download, transform=transform)
    
    # Return DataLoader if batch_size is specified
    if batch_size is not None:
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        return data_loader
    
    # Extract all data and convert to numpy
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))
    X, y = next(iter(data_loader))
    
    # Flatten images (from 28x28 to 784)
    X = X.view(X.size(0), -1).numpy()
    y = y.numpy()
    
    # Apply PCA if specified
    if pca_components is not None and pca_components < X.shape[1]:
        pca = PCA(n_components=pca_components, random_state=42)
        X = pca.fit_transform(X)
        print(f"Applied PCA: {X.shape[1]} dimensions, {sum(pca.explained_variance_ratio_)*100:.2f}% variance retained")
    
    return X, y

def visualize_means(means, image_shape=(28, 28), title="Cluster Means"):
    """
    Visualize the means of Gaussian components as images
    
    Args:
        means: Tensor or numpy array of shape (n_components, n_features)
        image_shape: Shape to reshape means to (for visualization)
        title: Plot title
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    
    if isinstance(means, torch.Tensor):
        means = means.cpu().numpy()
    
    n_components = means.shape[0]
    
    # Determine grid dimensions (try to make it square-ish)
    n_cols = int(np.ceil(np.sqrt(n_components)))
    n_rows = int(np.ceil(n_components / n_cols))
    
    fig = plt.figure(figsize=(n_cols * 2, n_rows * 2))
    grid = ImageGrid(fig, 111, nrows_ncols=(n_rows, n_cols), axes_pad=0.3)
    
    # Check if we need to reshape
    if means.shape[1] != np.prod(image_shape):
        # If means are in lower-dimensional space (e.g., after PCA)
        print(f"Warning: Means shape {means.shape[1]} doesn't match image size {np.prod(image_shape)}.")
        means_to_plot = means
    else:
        means_to_plot = means.reshape(n_components, *image_shape)
    
    for i, ax in enumerate(grid):
        if i < n_components:
            if len(means_to_plot.shape) == 2:
                # For lower-dimensional data, just show as a 1D signal
                ax.plot(means_to_plot[i])
                ax.set_title(f"Component {i}")
            else:
                # Show as image
                ax.imshow(means_to_plot[i], cmap='gray')
                ax.set_title(f"Component {i}")
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def plot_log_likelihood(log_likelihood_history, title="Log-Likelihood Convergence"):
    """
    Plot the log-likelihood history to visualize convergence
    
    Args:
        log_likelihood_history: List of log-likelihood values
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(log_likelihood_history)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Log-Likelihood")
    plt.grid(True)
    return plt.gcf()

def evaluate_clustering(y_true, y_pred, n_clusters=10):
    """
    Evaluate clustering performance by matching clusters to true labels
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster assignments
        n_clusters: Number of clusters/classes
        
    Returns:
        accuracy: Clustering accuracy after optimal matching
        confusion_matrix: Confusion matrix after matching
    """
    from sklearn.metrics import confusion_matrix
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(n_clusters))
    
    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Remap cluster labels
    y_pred_remapped = np.zeros_like(y_pred)
    for i, j in zip(col_ind, row_ind):
        y_pred_remapped[y_pred == i] = j
    
    # Compute accuracy
    accuracy = np.sum(y_pred_remapped == y_true) / len(y_true)
    
    # Recompute confusion matrix with optimal assignment
    remapped_cm = confusion_matrix(y_true, y_pred_remapped, labels=range(n_clusters))
    
    return accuracy, remapped_cm 