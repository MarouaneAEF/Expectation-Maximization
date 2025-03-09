import torch
import numpy as np
from tqdm import tqdm

class GaussianMixtureEM:
    """
    Gaussian Mixture Model using Expectation-Maximization algorithm
    """
    def __init__(self, n_components, n_features, init_means=None, init_covs=None, init_weights=None, 
                 tol=1e-5, max_iter=100, random_state=42):
        """
        Initialize GMM with EM algorithm
        
        Args:
            n_components: Number of Gaussian components
            n_features: Dimensionality of the data
            init_means: Initial means (if None, will be initialized randomly)
            init_covs: Initial covariance matrices (if None, will be initialized randomly)
            init_weights: Initial component weights (if None, will be initialized uniformly)
            tol: Convergence tolerance
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.n_features = n_features
        self.tol = tol
        self.max_iter = max_iter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Initialize parameters
        if init_means is None:
            self.means = torch.randn(n_components, n_features).to(self.device)
        else:
            self.means = init_means.to(self.device)
            
        if init_covs is None:
            # Initialize with identity covariance matrices
            self.covs = torch.stack([torch.eye(n_features) for _ in range(n_components)]).to(self.device)
        else:
            self.covs = init_covs.to(self.device)
            
        if init_weights is None:
            # Initialize with uniform weights
            self.weights = torch.ones(n_components).to(self.device) / n_components
        else:
            self.weights = init_weights.to(self.device)
            
        self.log_likelihood_history = []
        
    def _log_multivariate_normal_density(self, X, mean, cov):
        """
        Compute log probability of multivariate normal distribution
        
        Args:
            X: Data samples of shape (n_samples, n_features)
            mean: Mean vector of shape (n_features,)
            cov: Covariance matrix of shape (n_features, n_features)
            
        Returns:
            Log probabilities of shape (n_samples,)
        """
        n = X.shape[1]
        
        # Add small regularization to diagonal to ensure positive definiteness
        cov_reg = cov + torch.eye(n).to(self.device) * 1e-6
        
        try:
            # Compute determinant and inverse of covariance matrix
            L = torch.linalg.cholesky(cov_reg)
            log_det = 2 * torch.sum(torch.log(torch.diagonal(L)))
            L_inv = torch.inverse(L)
            cov_inv = L_inv.t() @ L_inv
            
            # Compute Mahalanobis distance (x-μ)ᵀΣ⁻¹(x-μ)
            X_centered = X - mean
            mahalanobis = torch.sum((X_centered @ cov_inv) * X_centered, dim=1)
            
            # Compute log pdf
            log_pdf = -0.5 * (n * np.log(2 * np.pi) + log_det + mahalanobis)
            return log_pdf
            
        except RuntimeError:
            # Fallback if Cholesky decomposition fails
            print("Warning: Cholesky decomposition failed. Using diagonal covariance.")
            var = torch.diagonal(cov_reg)
            log_det = torch.sum(torch.log(var))
            X_centered = X - mean
            mahalanobis = torch.sum(X_centered**2 / var, dim=1)
            log_pdf = -0.5 * (n * np.log(2 * np.pi) + log_det + mahalanobis)
            return log_pdf
            
    def _e_step(self, X):
        """
        Expectation step: compute responsibilities
        
        Args:
            X: Data samples of shape (n_samples, n_features)
            
        Returns:
            Responsibilities of shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        resp = torch.zeros(n_samples, self.n_components).to(self.device)
        
        # Compute log responsibilities for each component
        for k in range(self.n_components):
            resp[:, k] = torch.log(self.weights[k] + 1e-10) + self._log_multivariate_normal_density(
                X, self.means[k], self.covs[k]
            )
            
        # Log-sum-exp trick for numerical stability
        log_norm = torch.logsumexp(resp, dim=1, keepdim=True)
        log_resp = resp - log_norm
        resp = torch.exp(log_resp)
        
        # Compute log-likelihood
        log_likelihood = torch.sum(log_norm)
        
        return resp, log_likelihood
        
    def _m_step(self, X, resp):
        """
        Maximization step: update parameters
        
        Args:
            X: Data samples of shape (n_samples, n_features)
            resp: Responsibilities of shape (n_samples, n_components)
            
        Returns:
            Updated means, covariances, and weights
        """
        n_samples = X.shape[0]
        
        # Compute effective number of points per component
        resp_sum = torch.sum(resp, dim=0)
        
        # Update weights
        new_weights = resp_sum / n_samples
        
        # Update means
        new_means = torch.matmul(resp.t(), X) / resp_sum.unsqueeze(1)
        
        # Update covariances
        new_covs = torch.zeros_like(self.covs)
        for k in range(self.n_components):
            # Center data
            X_centered = X - new_means[k]
            
            # Compute weighted covariance
            cov_k = X_centered.t() @ (resp[:, k:k+1] * X_centered)
            cov_k /= resp_sum[k]
            
            # Ensure covariance is positive definite
            new_covs[k] = cov_k + torch.eye(self.n_features).to(self.device) * 1e-6
            
        return new_means, new_covs, new_weights
        
    def fit(self, X):
        """
        Fit the model to the data using EM algorithm
        
        Args:
            X: Data samples of shape (n_samples, n_features)
            
        Returns:
            self
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        elif isinstance(X, torch.Tensor):
            X = X.float().to(self.device)
            
        n_samples = X.shape[0]
        prev_log_likelihood = None
        
        # EM algorithm iterations
        for iteration in tqdm(range(self.max_iter), desc="EM iterations"):
            # E-step
            resp, log_likelihood = self._e_step(X)
            self.log_likelihood_history.append(log_likelihood.item())
            
            # Check for convergence
            if prev_log_likelihood is not None:
                change = abs(log_likelihood - prev_log_likelihood)
                if change < self.tol * abs(prev_log_likelihood):
                    print(f"Converged at iteration {iteration}")
                    break
                    
            prev_log_likelihood = log_likelihood
            
            # M-step
            self.means, self.covs, self.weights = self._m_step(X, resp)
            
        return self
        
    def predict(self, X):
        """
        Predict cluster labels for each sample
        
        Args:
            X: Data samples of shape (n_samples, n_features)
            
        Returns:
            Cluster assignments of shape (n_samples,)
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        elif isinstance(X, torch.Tensor):
            X = X.float().to(self.device)
            
        resp, _ = self._e_step(X)
        return torch.argmax(resp, dim=1).cpu().numpy()
        
    def score_samples(self, X):
        """
        Compute the log-likelihood of each sample
        
        Args:
            X: Data samples of shape (n_samples, n_features)
            
        Returns:
            Log-likelihood for each sample
        """
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float().to(self.device)
        elif isinstance(X, torch.Tensor):
            X = X.float().to(self.device)
            
        n_samples = X.shape[0]
        log_prob = torch.zeros(n_samples, self.n_components).to(self.device)
        
        for k in range(self.n_components):
            log_prob[:, k] = torch.log(self.weights[k] + 1e-10) + self._log_multivariate_normal_density(
                X, self.means[k], self.covs[k]
            )
            
        return torch.logsumexp(log_prob, dim=1).cpu().numpy() 