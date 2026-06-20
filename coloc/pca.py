import os
import joblib
from sklearn.decomposition import PCA

class EmbeddingPCA:
    """Wrapper around scikit-learn PCA to manage dimensionality reduction of patch embeddings."""
    def __init__(self, n_components=64):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def fit(self, X):
        """Fits the PCA model on the input features matrix X."""
        print(f"Fitting PCA model with {self.n_components} components...")
        self.pca.fit(X)
        return self

    def transform(self, X):
        """Projects the high-dimensional features matrix X onto the principal components."""
        return self.pca.transform(X)

    def fit_transform(self, X):
        """Fits the PCA model on X and returns the projected features."""
        print(f"Fitting and transforming PCA model with {self.n_components} components...")
        return self.pca.fit_transform(X)

    def save(self, file_path):
        """Persists the fitted PCA model using joblib."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(self.pca, file_path)
        print(f"PCA model saved to {file_path}")

    def load(self, file_path):
        """Loads a persisted PCA model from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No PCA model found at: {file_path}")
        self.pca = joblib.load(file_path)
        self.n_components = self.pca.n_components
        print(f"PCA model loaded from {file_path} (Components: {self.n_components})")
        return self
