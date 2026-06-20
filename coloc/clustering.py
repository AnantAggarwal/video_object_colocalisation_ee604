import os
import joblib
import numpy as np
from sklearn.cluster import KMeans

class FeatureClustering:
    """Wrapper around scikit-learn K-Means for unsupervised patch-level semantic clustering."""
    def __init__(self, n_clusters=16, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)

    def fit(self, X, subset_ratio=0.1):
        """
        Fits the K-Means clustering algorithm on a randomly sampled subset of feature matrix X
        to speed up computation and reduce RAM requirements.
        """
        n_samples = X.shape[0]
        sample_size = int(n_samples * subset_ratio)
        
        if sample_size < self.n_clusters:
            sample_size = n_samples
            
        print(f"Fitting K-Means (K={self.n_clusters}) on a {subset_ratio * 100:.1f}% subset ({sample_size}/{n_samples} patches)...")
        
        # Select random indices without replacement
        np.random.seed(self.random_state)
        subset_indices = np.random.choice(n_samples, sample_size, replace=False)
        X_subset = X[subset_indices]
        
        self.kmeans.fit(X_subset)
        return self

    def predict(self, X):
        """Predicts the cluster IDs for the input feature vectors X."""
        return self.kmeans.predict(X)

    def save(self, file_path):
        """Persists the fitted K-Means model using joblib."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.kmeans, file_path)
        print(f"K-Means model saved to {file_path}")

    def load(self, file_path):
        """Loads a persisted K-Means model from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No K-Means model found at: {file_path}")
        self.kmeans = joblib.load(file_path)
        self.n_clusters = self.kmeans.n_clusters
        print(f"K-Means model loaded from {file_path} (Clusters: {self.n_clusters})")
        return self
