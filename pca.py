#week1.py

import math
import pickle
import gzip
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import pytest
#%matplotlib inline
from sklearn.decomposition import PCA as pca1

import unittest

from sklearn.preprocessing import StandardScaler

class PCA:
    def __init__(self, target_explained_variance=None):
        """
        explained_variance: float, the target level of explained variance
        """
        self.target_explained_variance = target_explained_variance
        self.feature_size = -1

    def standardize(self, X):
        """
        standardize features using standard scaler
        :param X: input data with shape m (# of observations) X n (# of features)
        :return: standardized features (Hint: use skleanr's StandardScaler. Import any library as needed)
        """
        # your code here
        s_scaler = StandardScaler()
        # transform the dataset and returned scaled version
        x_scaled = s_scaler.fit_transform(X)
        return x_scaled
        

    def compute_mean_vector(self, X_std):
        """
        compute mean vector
        :param X_std: transformed data
        :return n X 1 matrix: mean vector
        """
        # your code here
        # Number of observations (by rows)
        #make sure nparray
        X_std = np.array(X_std)
        # Mean of each feature (column)
        mean_X = np.mean(X_std, axis=0)
        
        return mean_X
        

    def compute_cov(self, X_std, mean_vec):
        """
        Covariance using mean, (don't use any numpy.cov)
        :param X_std:
        :param mean_vec:
        :return n X n matrix:: covariance matrix
        """

        #number of observations (again)
        #make sure nparrays
        X_std = np.array(X_std)
        mean_vec = np.array(mean_vec)
        n = X_std.shape[0]
        # center the matrix by subtracting the mean of each column from all elements in the column
        X_centered = X_std - mean_vec
        # calculate the covariance matrix
        covariance_matrix = (X_centered.T @ X_centered) / (n - 1)
        return covariance_matrix

    def compute_eigen_vector(self, cov_mat):
        """
        Eigenvector and eigen values using numpy. Uses numpy's eigenvalue function
        :param cov_mat:
        :return: (eigen_values, eigen_vector)
        """
        #make sure nparray
        cov_mat = np.array(cov_mat)
        e_values, e_vectors = np.linalg.eigh(cov_mat)
        return e_values, e_vectors

    def compute_explained_variance(self, eigen_vals):
        """
        sort eigen values and compute explained variance.
        explained variance informs the amount of information (variance)
        can be attributed to each of  the principal components.
        :param eigen_vals:
        :return: explained variance.
        """
        
        sorted_evals = np.sort(eigen_vals)[::-1]
        total_var = np.sum(sorted_evals)
        explained_var_ratios = sorted_evals/total_var
        return explained_var_ratios
        

    def cumulative_sum(self, var_exp):
        """
        return cumulative sum of explained variance.
        :param var_exp: explained variance
        :return: cumulative explained variance
        """
        return np.cumsum(var_exp)

    def compute_weight_matrix(self, eig_pairs, cum_var_exp):
        """
        compute weight matrix of top principal components conditioned on target
        explained variance.
        (Hint : use cumulative explained variance and target_explained_variance to find
        top components)
        
        :param eig_pairs: list of tuples containing eigenvalues and eigenvectors, 
        sorted by eigenvalues in descending order (the biggest eigenvalue and corresponding eigenvectors first).
        :param cum_var_exp: cumulative expalined variance by features
        :return: weight matrix (the shape of the weight matrix is n X k)
        """
        # your code here
        
        ## solution
        # def compute_weight_matrix(self, eig_pairs, cum_var_exp):
        # """
        # compute weight matrix of top principal components conditioned on target
        # explained variance.
        # (Hint : use cumilative explained variance and target_explained_variance to find
        # top components)
        
        # :param eig_pairs: list of tuples containing eigenvalues and eigenvectors, 
        # sorted by eigenvalues in descending order (the biggest eigenvalue and corresponding eigenvectors first).
        # :param cum_var_exp: cumulative expalined variance by features
        # :return: weight matrix (the shape of the weight matrix is n X k)
        # """
        # ### BEGIN SOLUTION
        # matrix_w = np.ones((self.feature_size, 1))
        # for i in range(len(eig_pairs)):
        #     if cum_var_exp[i] < self.target_explained_variance:
        #         matrix_w = np.hstack((matrix_w,
        #                               eig_pairs[i][1].reshape(self.feature_size,
        #                                                       1)))
        # return np.delete(matrix_w, [0], axis=1).tolist()
        # ### END SOLUTION
        
        # extract eigenvalues
        # eig_values = np.array([e_pair[0] for e_pair in eig_pairs])
        # explained_variances = self.compute_explained_variance(eigen_vals=eig_values)
        # get target number of components based on the traget explained variance setting

        # make sure cum_var_exp is a numpy array
        cum_var_exp = np.array(cum_var_exp)
        
        # check to make sure there is at least one cumulative value higher than the target
        if np.max(cum_var_exp) < self.target_explained_variance:
            # If not, take everything
            num_components = len(cum_var_exp)
        else:
            num_components = np.argmax(cum_var_exp >= self.target_explained_variance) + 1
        # extract the eigenvectors for the top target components
        eig_vectors = np.array([pair[1] for pair in eig_pairs])
        top_components = np.array(eig_vectors[:num_components,:])
        # create the weight matrix by stacking the eigenvectors horizontally
        weight_matrix = top_components.T
        return weight_matrix
        

    def transform_data(self, X_std, matrix_w):
        """
        transform data to subspace using weight matrix
        :param X_std: standardized data
        :param matrix_w: weight matrix
        :return: data in the subspace
        """
        return X_std.dot(matrix_w)

    def fit(self, X):
        """    
        entry point to the transform data to k dimensions
        standardize and compute weight matrix to transform data.
        The fit functioin returns the transformed features. k is the number of features which cumulative 
        explained variance ratio meets the target_explained_variance.
        :param   m X n dimension: train samples
        :return  m X k dimension: subspace data. 
        """
    
        self.feature_size = X.shape[1]
        
        # your code here
        matrix_w = []
        
        ## SOLUTION
        # def fit(self, X):
        # """    
        # entry point to the transform data to k dimensions
        # standardize and compute weight matrix to transform data.
        # The fit functioin returns the transformed features. k is the number of features which cumulative 
        # explained variance ratio meets the target_explained_variance.
        # :param   m X n dimension: train samples
        # :return  m X k dimension: subspace data. 
        # """
    
        # self.feature_size = X.shape[1]
        
        # ### BEGIN SOLUTION
        # # 16 pts
        # X_std = self.standardize(X) # partial: 2 pts
        # #---- partial 2 pts
        # mean_vec = self.compute_mean_vector(X_std)
        # cov_mat = self.compute_cov(X_std, mean_vec) 
        # #-------
        # eig_vals, eig_vecs = self.compute_eigen_vector(cov_mat) #partial 2pts
        # #----- partial 4 pts
        # eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in
        #              range(len(eig_vals))]
        # eig_pairs.sort()
        # eig_pairs.reverse()
        # #-------
        # var_exp = self.compute_explained_variance(eig_vals) # partial 2 pts
        # cum_var_exp = self.cumulative_sum(var_exp) #partial 2pts
        # matrix_w = self.compute_weight_matrix(eig_pairs=eig_pairs,cum_var_exp=cum_var_exp) #partial 2 pts
        # ### END SOLUTION
        # print(len(matrix_w),len(matrix_w[0]))
        # return self.transform_data(X_std=X_std, matrix_w=matrix_w)
        
        
        #         The gist of PCA Algorithm to compute principal components is follows:

        # Calculate the covariance matrix X of data points.
        # Calculate eigenvectors and corresponding eigenvalues.
        # Sort the eigenvectors according to their eigenvalues in decreasing order.
        # Choose first k eigenvectors which satisfies target explained variance.
        # Transform the original data of shape m observations times n features into m observations times k selected features.
        X_std = self.standardize(X) #standardize and center
        mean_vec = self.compute_mean_vector(X_std=X_std)
        cov_matrix = self.compute_cov(X_std=X_std,mean_vec=mean_vec)
        eig_values, eig_vecs = self.compute_eigen_vector(cov_mat=cov_matrix)
        explained_variance = self.compute_explained_variance(eigen_vals=eig_values)
        cum_sum_var = self.cumulative_sum(var_exp=explained_variance)
        
        # Adjusting the eigenvectors (loadings) that are largest in absolute value to be positive (one method)
        max_abs_idx = np.argmax(np.abs(eig_vecs), axis=0)
        ev_range_top = eig_vecs.shape[0]
        signs = np.sign(eig_vecs[max_abs_idx, range(ev_range_top)])
        eig_vecs = eig_vecs*signs[np.newaxis,:]
        eig_vecs = eig_vecs.T
        
        # # method 2 of sign adjustment
        
        # # sort first
        # # Sort eigenvectors by eigenvalues in descending order
        # idx = np.argsort(eig_values)[::-1]
        # eig_values = eig_values[idx]
        # eig_vecs = eig_vecs[:, idx]
        
        # for i in range(eig_vecs.shape[1]):
        #     if eig_vecs[0, i] < 0:
        #         eig_vecs[:, i] = -eig_vecs[:, i]
        
        #sort by eigenvalues
        #make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_values[i]), eig_vecs[i,:]) for i in range(len(eig_values))]
        #sort the tuples from the highest to the lowest based on eigenvalues
        eig_pairs.sort(key=lambda x: x[0], reverse=True)
        
        #get matrix weights
        matrix_w = self.compute_weight_matrix(eig_pairs=eig_pairs, cum_var_exp=cum_sum_var)
        
        print(len(matrix_w),len(matrix_w[0]))
        return self.transform_data(X_std=X_std, matrix_w=matrix_w)


class test_PCA(unittest.TestCase):
    def setUp(self):
        self.PCA = PCA()
        self.PCA.target_explained_variance = 0.85 #set default starting point
        
    def test_scale_features(self):
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
                      
        expected_scaler = StandardScaler()
        expected_result = expected_scaler.fit_transform(X)
        
        scaled_X = self.PCA.standardize(X)
        
        # Using numpy.testing.assert_array_almost_equal to compare arrays
        np.testing.assert_array_almost_equal(scaled_X, expected_result, decimal=6, err_msg="The scaled features do not match the expected result.")
    
    def test_compute_eigen_vector(self):
        # Define a simple matrix
        matrix = np.array([[2, 0], [0, 3]])
        expected_eigenvalues = np.array([2, 3])
        expected_eigenvectors = np.array([[1, 0], [0, 1]])
        
        # Calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = self.PCA.compute_eigen_vector(matrix)
        
        # Assert that the calculated values are as expected
        self.assertTrue(np.allclose(eigenvalues, expected_eigenvalues))
        self.assertTrue(np.allclose(np.abs(eigenvectors), np.abs(expected_eigenvectors)))
        
    def test_compute_mean_vector(self):
        # rest data
        X = np.array([[1, 2, 3],
                      [5, 6, 7],
                      [9, 10, 11]])
        # expected result calculated with an alternative method
        mean_of_features = np.mean(X, axis=0)
        test_mean = self.PCA.compute_mean_vector(X_std=X)
        #centered_X = X - mean_of_features
        self.assertTrue(np.allclose(mean_of_features, test_mean),
                        "The means of the numpy.cov calc and result are not close enough.")
        
    def test_compute_cov(self):
        #test data
        X = np.array([[1, 2, 3],
                      [5, 6, 7],
                      [9, 10, 11]])
        # Expected result calculated manually or with an alternative method
        expected_cov_matrix = np.cov(X, rowvar=False)
        expected_mean = np.mean(X, axis=0)
        # Calculate the covariance matrix using the function
        calculated_cov_matrix = self.PCA.compute_cov(X_std=X, mean_vec=expected_mean)
        
        # Assert that the calculated covariance matrix matches the expected result
        # np.allclose is used to compare floating-point matrices
        self.assertTrue(np.allclose(calculated_cov_matrix, expected_cov_matrix),
                        "The calculated covariance matrix does not match the expected result.")
    
    def test_compute_explained_variance(self):
        # Example eigenvalues
        eigenvalues = np.array([0.5, 1.5, 0.2, 0.8])
        # Expected explained variance ratio, calculated manually or with another method
        expected_ratio = np.array([1.5, 0.8, 0.5, 0.2]) / np.sum([0.5, 1.5, 0.2, 0.8])

        # Calculate explained variance ratio using the function
        calculated_ratio = self.PCA.compute_explained_variance(eigenvalues)

        # Check if the calculated ratio matches the expected ratio
        np.testing.assert_array_almost_equal(calculated_ratio, expected_ratio,
                                             decimal=6, err_msg="Explained variance ratio does not match expected values.")

    def test_compute_weight_matrix(self):
        # Define test eigenpairs (eigenvalue, eigenvector)
        eig_pairs = [
            (4, np.array([0.5, 0.5, 0.7])),
            (2, np.array([0.2, 0.8, 0.1])),
            (1, np.array([0.9, 0.1, 0.3]))
        ]
        # Target explained variance
        target_explained_variance = 0.85
        self.PCA.target_explained_variance = target_explained_variance

        # Expected number of components to reach target variance
        expected_num_components = 2

        # Expected weight matrix shape
        expected_shape = (3, expected_num_components)  # 3 features, 2 top components
        
        # Extract the eigenvalues and calculate the total variance
        eigenvalues = np.array([pair[0] for pair in eig_pairs])
        total_variance = np.sum(eigenvalues)
    
        # Calculate the explained variance ratios for each eigenvalue
        explained_variances = eigenvalues / total_variance
    
        # Calculate cumulative explained variance
        cum_var_exp = np.cumsum(explained_variances)

        # Compute weight matrix using the function
        weight_matrix = self.PCA.compute_weight_matrix(eig_pairs, cum_var_exp)

        # Check if the weight matrix has the correct shape
        self.assertEqual(weight_matrix.shape, expected_shape, "Weight matrix does not have the expected shape.")

        # Additional checks could include verifying the content of the weight matrix
        # For this, we need the expected weight matrix, which is the stack of the top eigenvectors
        expected_weight_matrix = np.vstack([eig_pairs[0][1], eig_pairs[1][1]]).T
        np.testing.assert_array_almost_equal(weight_matrix, expected_weight_matrix, decimal=6,
                                             err_msg="Weight matrix does not match the expected matrix.")

    def test_fit_shapes(self):
        try:
            X = np.random.rand(100, 10) 
            self.PCA.target_explained_variance = 0.80
            transformed_data = self.PCA.fit(X)
            self.assertEqual(transformed_data.shape[0], X.shape[0])  # Check if the number of samples is the same
            self.assertTrue(transformed_data.shape[1] <= X.shape[1])  # Check if the number of features is reduced
        except Exception as e:
            self.fail(f"Test failed with exception: {e}")
    
    def test_fit_against_sklearn(self):
        try:
            #set target explained variance
            target_explained_variance = 0.9
            # Create a random dataset
            X = np.random.rand(100, 10)
            # Fit the PCA model
            self.PCA.target_explained_variancee = target_explained_variance
            transformed_data = self.PCA.fit(X)
            # Compare the result with sklearn's PCA
            from sklearn.decomposition import PCA as SKPCA
            sk_pca = SKPCA(n_components=target_explained_variance)
            sk_transformed_data = sk_pca.fit_transform(X)
            #test if both selected the same number of components
            # np.testing.assert_array_almost_equal(transformed_data, sk_transformed_data, decimal=6,
            #                                      err_msg="The transformed data does not match the expected result.")
            self.assertAlmostEqual(transformed_data.shape[1], sk_transformed_data.shape[1], delta=1,
                                                msg="The transformed data does not match the sklearn in number of components chosen.")
        except Exception as e:
            self.fail(f"Test failed with exception: {e}")
            
    def test_course_standardize(self):
        np.random.seed(42)
        X = np.array([[0.39, 1.07, 0.06, 0.79], [-1.15, -0.51, -0.21, -0.7], [-1.36, 0.57, 0.37, 0.09], [0.06, 1.04, 0.99, -1.78]])
        pca_handler = self.PCA
        pca_handler.target_explained_variance = 0.99
        X_std_act = pca_handler.standardize(X)

        X_std_exp = [[ 1.20216033, 0.82525828, -0.54269609, 1.24564656],
                    [-0.84350476, -1.64660539, -1.14693504, -0.31402854],
                    [-1.1224591, 0.04302294, 0.15105974, 0.51291329],
                    [ 0.76380353, 0.77832416, 1.53857139, -1.4445313]]

        for act, exp in zip(X_std_act, X_std_exp):
            assert pytest.approx(act, 0.01) == exp, "Check Standardize function"
         
    def test_course_compute_mean_vector(self):
        np.random.seed(42)
        pca_handler = self.PCA
        pca_handler.target_explained_variance = 0.99   
        X_std_exp = [[ 1.20216033, 0.82525828, -0.54269609, 1.24564656],
                    [-0.84350476, -1.64660539, -1.14693504, -0.31402854],
                    [-1.1224591, 0.04302294, 0.15105974, 0.51291329],
                    [ 0.76380353, 0.77832416, 1.53857139, -1.4445313]]
        
        mean_vec_act = pca_handler.compute_mean_vector(X_std_exp)
        mean_vec_exp = [5.55111512, 2.77555756, 5.55111512, -5.55111512]
        mean_vec_act_tmp = mean_vec_act * 1e17
        
        print ((1.24564656+-0.31402854 + 0.51291329 -1.4445313)/4)

        assert pytest.approx(mean_vec_act_tmp, 0.1) == mean_vec_exp, "Check compute_mean_vector function"
        
    def test_course_compute_cov(self):
        pca_handler = self.PCA
        pca_handler.target_explained_variance = 0.99
        X_std_exp = [[ 1.20216033, 0.82525828, -0.54269609, 1.24564656],
                    [-0.84350476, -1.64660539, -1.14693504, -0.31402854],
                    [-1.1224591, 0.04302294, 0.15105974, 0.51291329],
                    [ 0.76380353, 0.77832416, 1.53857139, -1.4445313]]
        mean_vec_exp = [5.55111512, 2.77555756, 5.55111512, -5.55111512]
        #mean_vec = np.array(mean_vec_exp) * -1e17
        mean_vec = pca_handler.compute_mean_vector(X_std_exp)
            
        cov_mat_act = pca_handler.compute_cov(X_std_exp, mean_vec) 

        cov_mat_exp = [[ 1.33333333, 0.97573583, 0.44021511, 0.02776305],
        [ 0.97573583, 1.33333333, 0.88156376, 0.14760488],
        [ 0.44021511, 0.88156376, 1.33333333, -0.82029039],
        [ 0.02776305, 0.14760488, -0.82029039, 1.33333333]]

        assert pytest.approx(cov_mat_act, 0.01) == cov_mat_exp, "Check compute_cov function"
    
    def test_course_eigen_vector(self):
        np.random.seed(42)
        pca_handler = self.PCA
        pca_handler.target_explained_variance = 0.99
        cov_mat_exp = [[ 1.33333333, 0.97573583, 0.44021511, 0.02776305],
        [ 0.97573583, 1.33333333, 0.88156376, 0.14760488],
        [ 0.44021511, 0.88156376, 1.33333333, -0.82029039],
        [ 0.02776305, 0.14760488, -0.82029039, 1.33333333]]  
        eig_vals_act, eig_vecs_act = pca_handler.compute_eigen_vector(cov_mat_exp) 

        eig_vals_exp = [2.96080083e+00, 1.80561744e+00, 5.66915059e-01, 7.86907276e-17]

        eig_vecs_exp = [[ 0.50989282,  0.38162981,  0.72815056,  0.25330765],
        [ 0.59707545,  0.33170546, -0.37363029, -0.62759286],
        [ 0.57599397, -0.37480162, -0.41446394,  0.59663585],
        [-0.22746684,  0.77708038, -0.3980161,   0.43126337]]

        assert pytest.approx(eig_vals_act, 0.01) == eig_vals_exp, "Check compute_eigen_vector function eigen values"

        for act, exp in zip(eig_vecs_act, eig_vecs_exp):
            assert pytest.approx(act, 0.01) == exp, "Check compute_eigen_vector function eigen vectors"
    
    def test_course_compute_explained_variance(self):
        pca_handler = self.PCA
        pca_handler.target_explained_variance = 0.99
        X = np.array([[0.39, 1.07, 0.06, 0.79], [-1.15, -0.51, -0.21, -0.7], [-1.36, 0.57, 0.37, 0.09], [0.06, 1.04, 0.99, -1.78]])
                
        pca_handler.feature_size = X.shape[1]
        eig_vals_exp = [2.96080083e+00, 1.80561744e+00, 5.66915059e-01, 7.86907276e-17]
        var_exp_act = pca_handler.compute_explained_variance(eig_vals_exp) 

        var_exp_exp = [0.5551501556710813, 0.33855327084133857, 0.10629657348758019, 1.475451142706682e-17]

        assert pytest.approx(var_exp_act, 0.01) == var_exp_exp, "Check compute_explained_variance function"
        
    def test_course_matrix_weights(self):
        np.random.seed(42)
        self.PCA.target_explained_variance = 0.99
        self.PCA.feature_size = 4 #set feature size based on test data  
        eig_pairs = np.array([(2.9608008302457662, np.array([ 0.50989282,  0.59707545,  0.57599397, -0.22746684])),
        (1.8056174444871387, np.array([ 0.38162981,  0.33170546, -0.37480162,  0.77708038])),
        (0.5669150586004276, np.array([ 0.72815056, -0.37363029, -0.41446394, -0.3980161 ])), 
        (7.869072761102302e-17, np.array([ 0.25330765, -0.62759286,  0.59663585,  0.43126337]))], dtype=object)

        cum_var_exp = np.array([0.55515016, 0.89370343, 1, 1])

        matrix_w_exp = np.array([[0.50989282, 0.38162981], 
                        [0.59707545, 0.33170546], 
                        [0.57599397, -0.37480162], 
                        [-0.22746684, 0.77708038]])

        matrix_w_act = self.PCA.compute_weight_matrix(eig_pairs=eig_pairs, cum_var_exp=cum_var_exp)

        for act, exp in zip(matrix_w_act, matrix_w_exp):
            assert pytest.approx(act, 0.001) == exp, "Check compute_weight_matrix function"
    
    def test_course_test_against_sklearn(self):
        #start with a comparison set
        pca_handler = self.PCA
        pca_handler.target_explained_variance = 0.85
        
        X_std_exp = np.array([[ 1.20216033, 0.82525828, -0.54269609, 1.24564656],
                    [-0.84350476, -1.64660539, -1.14693504, -0.31402854],
                    [-1.1224591, 0.04302294, 0.15105974, 0.51291329],
                    [ 0.76380353, 0.77832416, 1.53857139, -1.4445313]])
        
        matrix_w_exp = np.array([[0.50989282, 0.38162981], 
                        [0.59707545, 0.33170546], 
                        [0.57599397, -0.37480162], 
                        [-0.22746684, 0.77708038]])
        
        X = np.array([[0.39, 1.07, 0.06, 0.79], [-1.15, -0.51, -0.21, -0.7], [-1.36, 0.57, 0.37, 0.09], [0.06, 1.04, 0.99, -1.78]])

    
        pca_dev_result_transform_only = pca_handler.transform_data(X_std=X_std_exp, matrix_w=matrix_w_exp) 
        
        #go through full process
        pca_dev_result_full = pca_handler.fit(X)
        
        # Scale data before applying PCA
        scaling=StandardScaler()
        
        # Use fit and transform method
        # You may change the variable X if needed to verify against a different dataset
        print("Sample data:", X)
        scaling.fit(X)
        Scaled_data=scaling.transform(X)
        print("\nScaled data:", Scaled_data)
        
        # Set the n_components= target size
        principal=pca1(n_components=2)
        #principal = pca1(n_components=pca_handler.target_explained_variance)
        principal.fit(Scaled_data)
        pca_sklearn_result = principal.transform(Scaled_data)
        
        # Check the dimensions of data after PCA
        print("\nTransformed Data",pca_sklearn_result) 
        
        assert pytest.approx(pca_dev_result_full, 0.01) == pca_dev_result_transform_only, "Expected result from transform only does not match full pca fit result"     
        
        assert pytest.approx(pca_dev_result_full, 0.01) == pca_sklearn_result, "Expected result from sklearn pca does not match full pca fit result" 
        
if __name__ == '__main__':
    unittest.main()