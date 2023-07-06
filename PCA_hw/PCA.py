import numpy as np


class myPCA:
    def __init__(self, n_components: int = 2, method: str = 'svd') -> None:
        '''
        The constructor of the PCA algorithm.

        :param n_components: int, default=2
            The dimension to which the data will be reduced.
        :param method: str, default='svd'
            The method used by PCA to reduce the dimensionality of the data.
        '''
        self.__n_components = n_components
        if method in ['svd', 'eigen']:
            self.__method = method
        else:
            raise ValueError(
                f"'{method}' is not a method implemented in this model")

    def fit(self, X: 'np.array'):
        '''
        The fitting method.

        :param X: np.array
            The data on which we want to fit the PCA
        '''
        if self.__method == 'svd':
            U, S, V = np.linalg.svd(X)
            self.__V = V[:self.__n_components, :]
        elif self.__method == 'eigen':
            corr_mat = np.corrcoef(X.T)
            # Getting the eigenvectors and eigenvalues
            self.eig_vals, self.eig_vecs = np.linalg.eig(corr_mat)

            # Sorting the list of tuples (eigenvalue, eigenvector)
            self.eig_pairs = [(np.abs(self.eig_vals[i]), self.eig_vecs[:, i])
                              for i in range(len(self.eig_vals))]
            self.eig_pairs.sort(key=lambda x: x[0], reverse=True)

            # Calculating the explained ratio
            total = sum(self.eig_vals)
            self.explained_variance_ratio = [
                (i / total) * 100 for i in sorted(self.eig_vals, reverse=True)]
            self.cumulative_variance_ratio = np.cumsum(
                self.explained_variance_ratio)

            # Creating the projection matrix
            self.matrix_w = np.hstack((self.eig_pairs[i][1].reshape(
                np.size(X, 1), 1) for i in range(self.__n_components)))
        return self

    def transform(self, X: 'np.array') -> 'np.array':
        '''
        The transform function.

        :param X: np.array
            The data that we must reduce.
        '''
        if self.__method == 'svd':
            return X.dot(self.__V.T)
        elif self.__method == 'eigen':
            return X.dot(self.matrix_w)