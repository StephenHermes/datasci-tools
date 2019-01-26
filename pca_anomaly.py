"""PCA anomaly detection a la Shyu, Chen, Sarinnaparkorn and Chang.

A function and an scikit-learn style anomaly detector based off of 
Shyu, Chen, Sarinnapakorn and Chang's paper 'A Novel Anomaly Detection 
Scheme Based on Prinicpal Component Classifier'. 

The scheme has three steps:
1. Get a robust representation of the input data X by trimming
    extreme observatons in terms of the Mahalanobis distance from the
    mean. Done with the `trim` function.
2. Train a classifier off of the robust representative. The classifier
    is built using principal components analysis of the robust matrix.
    Done with a `PCADetector` object's `fit` method.
3. Categorize data as normal or anomalous. Done with a fitted 
    `PCADetector` object's `predict` method.

"""

# general imports
import numpy as np
# import for trimming function
from heapq import nsmallest

# imports for ML
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class MahalanobisTrimmer(BaseEstimator, TransformerMixin):
    """Mahalanobis distance based trimmer.

    Trim the most extreme elements of the data set X in terms of 
    Mahalanobis distance to the mean. Gamma determines proportion of 
    data to remove.

    Parameters
    ----------
    X : array of size (n_samples, n_features)
        The data to tim
    
    gamma : float
        The proportion of data to be trimmed.

    Returns
    -------
    X_trim : array of shape (n_samples, n_features)
        The data with the gamma-most extreme observations removed.
    """

    def __init__(self, gamma=0.005):
        self.gamma = gamma

    def fit(self, X, y=None):
        # number of rows to keep
        n_keep = int(X.shape[0] * (1 - self.gamma))
        self._n_keep = n_keep
        # get correlation matrix
        S = np.corrcoef(X.transpose())
        S_inv = np.linalg.inv(S)
        self._S_inv = S_inv
        # get means of features
        self.means_ = np.mean(X, axis=1)
        return self
        
    def transform(self, X, y=None):
        x_bar = self.means_
        S_inv = self._S_inv
        # Mahalanobis distance
        def dist_man(x): return (x - x_bar) @ S_inv @ (x - x_bar)
        # trimmed data
        X_trim = nsmallest(self._n_keep, X, key=dist_man)
        return np.array(X_trim)


class PCADetector(BaseEstimator):
    """Anomaly detection using PCA.

    A scikit-learn style anomaly detector based off of Shyu, Chen, 
    Sarinnapakorn and Chang's paper 'A Novel Anomaly Detection Scheme 
    Based on Prinicpal Component Classifier'. 
    
    The underscore conventions in names are not correct for sklearn.
    
    Parameters
    ----------
    alpha : float, optional (default=0.05) 
        Acceptable false positive rate.

    n_major : int (optional, default=None) 
        Number of principal components to use for major component, i.e., 
        components detecting general trend of the data. If None passed 
        then takes the first `n_major` principal components which account 
        for 85% of total variance.
    
    n_minor: int (optional, default=None)
        Number of principal components to use for minor component, i.e., 
        components detecting the general correlations between features.  
        If None, then uses all components which individually account for 
        less than 20% of the total variance.
    
    """    
    def __init__(self, alpha=0.05, n_major=None, n_minor=None):
        # public attributes
        self.alpha = alpha
        self.n_major = n_major
        self.n_minor = n_minor

    def _get_eigens(self, X_trim):
        """ Get correlation matrix and its eigenvector/values."""
        self._S = np.corrcoef(X_trim.transpose())
        self._eigvals, self._eigvects = np.linalg.eigh(self._S)
           
    def _get_comps(self, z):
        """Input vector z should be normalized.
        
        Parameters
        ----------
        z : array
            A normalized vector, with mean zero and standard deviation one.
        
        Returns
        -------
        major : array
            The  projection of a normalized vector z onto the major components
            determined by the PCA detector.

        minor : array
            The  projection of a normalized vector z onto the minor components
            determined by the PCA detector.

        """      
        y = self._eigvects @ z
        summands = y ** 2 / self._eigvals
        p_major = self.n_major
        p_minor = len(self._eigenvals) - self.n_minor
        major = summands[:self.p_major].sum()
        minor = summands[self.p_minor:].sum()
        return major, minor
    
    def fit(self, X_trim, y=None):
        """Fit the classifier to the trimmed data. """
        # get eigenvalues and vectors
        self._get_eigens(X_trim)
        # instantiate scaler for z-scores
        scaler = StandardScaler()
        scaler.fit(X_trim)
        self._scaler = scaler
        Z = scaler.transform(X_trim)
        # get major and minor components of each input vector
        self.majors_ = []
        self.minors_ = []
        for i in range(X_trim.shape[0]):
            major, minor = self._get_comps(Z[i, :])
            self.majors_.append(major)
            self.minors_.append(minor)
        # get cutoffs for classification
        c1 = np.percentile(self._majors, 100 * (1 - self.alpha))
        c2 = np.percentile(self._minors, 100 * (1 - self.alpha))
        self.c_ = (c1, c2)
        # autoset n_major and n_minor if not specifiec
        if (self.n_major is None) or (self.n_minor is None):
            var = self._eigenvals / self._eigenvals.sum()
            var = np.sort(variances)[::-1]
            if self.n_major is None:
                # set n_major to account for 85% of total variance
                self.n_major = 1 + (var.cumsum() < 0.85).sum()
            if self.n_minor is None:
                # set n_minor to components with < 20% variance
                n_minor = (var < 0.2).sum()
                n_minor_max = len(self._eignevals) - self.n_major 
                self.n_minor = min(n_minor_max, n_minor)
        return self

    def predict(self, X, y=None):
        """Classify as normal or anomalous.

        Returns
        -------
        cl : array
            Whether normal (1) or anomalous (-1).
        
        """
        Z = self._scaler.transform(X)
        cl = []
        for i in range(Z.shape[0]):
            z_major, z_minor = self._get_comps(Z[i, :])
            if (z_major < self.c_[0]) & (z_minor < self.c_[1]):
                cl.append(1)
            else:
                cl.append(-1)
        return np.array(cl)
