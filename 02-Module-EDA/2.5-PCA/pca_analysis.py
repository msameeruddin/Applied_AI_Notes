"""
Refer to → https://builtin.com/data-science/step-step-explanation-principal-component-analysis
"""

import numpy as np
import pandas as pd

from scipy.linalg import eigh

class PCA():
    def __init__(self, original_data, shrink_to=2):
        # self.shrink_to = shrink_to if (shrink_to == 2) else 2
        self.shrink_to = shrink_to
        self.original_data = original_data
        self.standardized_data = self.standardize_data_features()

    def standardize_data_features(self, in_numpy=False):
        dummy_data = self.original_data.copy()

        for col in dummy_data.columns:
            mean_val = np.mean(dummy_data[col])
            std_dev_val = np.std(dummy_data[col])
            dummy_data[col] = [(dval - mean_val)/std_dev_val for dval in dummy_data[col]]

        if not in_numpy:
            return dummy_data
        return dummy_data.to_numpy()

    def compute_covariance(self, X, Y):
        mean_x = np.mean(X)
        mean_y = np.mean(Y)

        X_ = X - mean_x
        Y_ = Y - mean_y
        cov_val = np.dot(X_, Y_)/(X.shape[0] - 1)

        return cov_val

    def compute_covariance_matrix(self):
        cm = [
            np.array([
                self.compute_covariance(
                    X=self.standardized_data[col_i], 
                    Y=self.standardized_data[col_j]
                ) for col_j in self.standardized_data.columns
            ]) 
            for col_i in self.standardized_data.columns
        ]
        return np.array(cm)

    def compute_eigen_vv(self):
        data_matrix = self.compute_covariance_matrix()

        evalues, evectors = eigh(a=data_matrix)
        evectors = evectors.T
        
        return evalues[-self.shrink_to:], evectors[-self.shrink_to:]

    def retain_important(self):
        _, vecs = self.compute_eigen_vv()

        stz_data = self.standardized_data.to_numpy()
        reduced_data = np.dot(vecs, stz_data.T)
        rdf = pd.DataFrame(data=reduced_data.T, columns=['PC' + str(i + 1) for i in range(reduced_data.shape[0])])
        
        return rdf


if __name__ == '__main__':
    size = (500, 20)
    cols = [str('col') + str(i) for i in range(size[1])]
    data = np.random.randint(low=5, high=20, size=size)
    data = pd.DataFrame(data=data, columns=cols)

    ###### code from sratch ######
    pca1 = PCA(original_data=data, shrink_to=2)
    rdf = pca1.retain_important()
    print(rdf.head())
    print(rdf.shape)
    print('================')
    ###### library code #####
    from sklearn import decomposition
    from sklearn.preprocessing import StandardScaler
    sample_data = StandardScaler().fit_transform(data)
    pca2 = decomposition.PCA()
    pca2.n_components = 2
    pca_data = pca2.fit_transform(sample_data)
    pca_df = pd.DataFrame(data=pca_data, columns=("PC1", "PC2"))
    print(pca_df.head())
    print(pca_df.shape)
    print('================')