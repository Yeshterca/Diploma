# IMPORTS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import nibabel as nib
import os
import numpy as np
from sklearn.cluster import KMeans


# LOAD DATA FOR ONE SUBJECT
def load_subjects(dirpath, filename, no_subjects):
    """
    Load beta .nii files of all subjects and prepare the data for further use.
    :param dirpath: full path to the data directory
    :param filename: file with beta values (beta_0001.nii), 3D matrix
    :param no_subjects: number of all subjects
    :return: betas_all_valid ... beta values of all subjects, flatten to [SUB x flatten file],
                                nan values replaced with zeros
    """
    betas_all = np.zeros((no_subjects, 91*109*91))
    i = 0
    for dire in os.listdir(dirpath):
        for file in os.listdir(dirpath + '\\' + dire):
            if file == filename:
                sub_dir = os.path.join(dirpath, dire, filename)
                betas_subject = nib.load(sub_dir).get_fdata()
                betas_subject_fl = betas_subject.flatten()
                betas_all[i, :] = betas_subject_fl
        i += 1

    betas_nan = np.isnan(betas_all)
    betas_all_valid = np.where(betas_nan, 0, betas_nan)

    return betas_all_valid


def prepare_data(betas_all):  # TODO check ... invalid value in divide + put to PCA fc
    """
    Standardize the data.
    :param betas_all: file with all beta values of all patients to be standardized
    :return: betas_st ... standardized data
    """
    betas_st = betas_all.copy()
    betas_st[betas_st.columns] = StandardScaler().fit_transform(betas_st)
    #betas_st = normalize(betas_all)

    return betas_st


def pca(betas_all, no_components):
    """
    Find low dim space the preserves the most variance of the data.
    :param betas_all: beta files of all subjects [SUB x 3d flatten]
    :param no_components: dimensionality, number of components
    :return: principal components ... basis vectors of new low dim space [3d flatten x no_components]
    """
    pca_VW = PCA(n_components=no_components)
    principal_components = pca_VW.fit_transform(betas_all.T)

    return principal_components


def pca_projection(betas_all_fl, principal_components, no_subjects, no_components):
    """
    Project the data of each subject to new low-dim space.
    :param betas_all_fl: data of all subjects (flatten) [SUB x 3d flatten]
    :param principal_components: basis vectors [3d flatten x no_components]
    :param no_subjects: number of all subjects
    :param no_components: number of components
    :return: features ... data projected to new low dim space
    """
    features = np.zeros((no_subjects, no_components,))
    for i in range(0, no_subjects):
        for j in range(0, no_components):
            features[i, j] = np.dot(betas_all_fl[i, :], principal_components[:, j])

    return features


def clustering(betas):
    kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto')
    kmeans.fit(betas)

    return kmeans


