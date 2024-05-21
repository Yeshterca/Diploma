import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn import metrics
from ica_run import *
from sklearn.linear_model import LogisticRegression
from skimage.segmentation import slic
from vw_run import *

def load_panss(dir):

    data = pd.read_excel(dir, header=None)
    selected_columns = data.iloc[1:, 36:40]   #36:40,  6:13
    selected_columns_num = selected_columns.apply(pd.to_numeric, errors='coerce')
    panss = selected_columns_num.values

    return panss  #(136, 4)


def exclude_inv_fov(panss_all, inv_subs, no_hc):

    panss_valid_all = np.delete(panss_all, inv_subs, axis=0)  #(131, 4)
    panss_fes = panss_valid_all[no_hc:]
    nan_idx = np.isnan(panss_fes[:, -1])
    panss_fes_valid = panss_fes[~nan_idx]  #(74, 4)
    panss_valid_all = np.where(np.isnan(panss_valid_all), 0, panss_valid_all)

    return panss_fes_valid, panss_valid_all, nan_idx


def load_VW_data(directory, filename, no_subjects, no_hc, nan_idx):
    betas_all = np.zeros((no_subjects, 91, 109, 91))
    i = 0
    for dire in os.listdir(directory):
        for file in os.listdir(directory + '\\' + dire):
            if file == filename:
                sub_dir = os.path.join(directory, dire, filename)
                betas_subject = nib.load(sub_dir).get_fdata()
                betas_all[i, :, :, :] = betas_subject
        i += 1

    betas_all = z_score_normalize_nudz(betas_all, no_hc)

    betas_fes = betas_all[no_hc:, :, :, :]
    features_vw = betas_fes[~nan_idx, :, :, :]

    mask = np.ones_like(features_vw[0, :, :, :])
    count_nans = np.zeros_like(features_vw[0, :, :, :])

    for i in range(0, np.shape(features_vw)[0]):
        nan_mask = np.isnan(features_vw[i, :, :, :])
        count_nans = count_nans + nan_mask
    mask[count_nans > 0] = 0

    betas_nan0 = features_vw
    betas_nanout = np.isnan(features_vw)
    betas_nan0[betas_nanout] = 0
    features_small = np.zeros_like(features_vw)

    for i in range(0, np.shape(features_vw)[0]):
        features_small[i, :, :, :] = betas_nan0[i, :, :, :] * mask

    features_large = np.where(features_small == 0., 10., features_small)


    return features_large, features_small


def load_ica_data(directory, no_subjects, no_hc, no_fes, relevant_ics, no_components, nan_idx):

    # load all
    component_numbers, betas_all = get_ica_data(directory)
    # select significant ics
    ic_numbers_selected, ic_betas_selected_all = select_components(no_subjects, component_numbers, relevant_ics, betas_all,
                                                               no_components)
    # get labels and weights
    ic_betas_selected = ic_betas_selected_all[:, no_hc:]
    ic_betas_selected = ic_betas_selected[:, ~nan_idx]

    # norm
    mean_fes = np.mean(ic_betas_selected.T)
    std_fes = np.std(ic_betas_selected.T)
    betas_fes_norm = (ic_betas_selected.T - mean_fes) / std_fes
    betas_all_norm = (ic_betas_selected_all.T - mean_fes) / std_fes

    features_ica_fes = betas_fes_norm  #.T
    features_ica_all = betas_all_norm  #.T
    labels = create_labels(no_hc, no_fes)

    return features_ica_fes, features_ica_all, labels

def regression_model(ica_features_train, panss_train, ica_features_val, kernel):

    regr = SVR(kernel=kernel)

    # training
    regr.fit(ica_features_train, panss_train)  # TODO select parameter / for all parameters

    # predict
    pred = regr.predict(ica_features_val)

    return pred


def LOO_svr_ica(features, panss, kernel):

    folds = len(panss)
    predictions = np.zeros((folds,))

    for fold in range(folds):
        features_train = np.delete(features, fold, 0)
        panss_train = np.delete(panss, fold)
        print(np.shape(features_train))

        features_val = features[fold, :]
        features_val = features_val.reshape(1, -1)
        print(np.shape(features_val))

        prediction = regression_model(features_train, panss_train, features_val, kernel)
        predictions[fold] = prediction[0]  #round(prediction[0])

    return predictions


def LOO_svr_vw(features_large, features_small, panss, num_segments, no_features, kernel):

    folds = len(panss)
    predictions = np.zeros((folds,))

    for fold in range(folds):
        print('FOLD: ', fold)
        features_train_large = np.delete(features_large, fold, 0)
        features_train_small = np.delete(features_small, fold, 0)
        panss_train = np.delete(panss, fold)

        features_val = features_small[fold, :, :, :]

        features_mean_large = mean_features(features_train_large)

        segments = slic(features_mean_large, num_segments, compactness=0.2, enforce_connectivity=True)

        pixel_means_train, pixel_means_all = vw_get_features_dimred_2d(features_train_small, segments, len(panss)-1)
        #pca = PCA(n_components=no_features)
        #pc_train = pca.fit_transform(pixel_means_train.T)
        pls_train, pls = get_features_pls(pixel_means_train.T, panss_train, no_features)

        features_valsubj_clustered = betas_in_superpixels_2d(features_val, segments)
        pixel_means_val = mean_superpixels(features_valsubj_clustered)
        #pixel_means_val = -1 + 2 * (pixel_means_val - np.min(pixel_means_all)) / (np.max(pixel_means_all) - np.min(pixel_means_train))
        pixel_means_val = pixel_means_val.reshape(1, -1)

        #pc_val = pca.transform(pixel_means_val)
        pls_val = pls.transform(pixel_means_val)

        prediction = regression_model(pls_train, panss_train, pls_val, kernel)
        predictions[fold] = prediction[0]

    return predictions


def betas_in_superpixels_2d(betas, segments):

    betas_sup = [[] for _ in range(np.max(segments)+1)]
    #coords = [[] for _ in range(np.max(segments)+1)]
    for (x, y), label in np.ndenumerate(segments):
        betas_sup[label].append(betas[x, y])
        #coords[label].append([x, y, z])
    betas_sup = [np.array(vals) for vals in betas_sup]
    #coords = [np.array(vals) for vals in coords]

    return betas_sup  #, coords


def vw_get_features_dimred_2d(betas_all, segments, no_subjects):

    for i in range(0, no_subjects):
        betas_subj_clustered = betas_in_superpixels_2d(betas_all[i, :, :, :], segments)
        if i == 0:
            pixel_means_all = np.zeros((len(betas_subj_clustered), no_subjects))
        pixel_means_all[:, i] = mean_superpixels(betas_subj_clustered)

    pixel_means_norm = pixel_means_all  #-1 + 2 * (pixel_means_all - np.min(pixel_means_all)) / (np.max(pixel_means_all) - np.min(pixel_means_all))

    return pixel_means_norm, pixel_means_all

def mean_features(features):

    features_mean = np.mean(features, axis=0)
    features_norm = features_mean  #(features_mean - features_mean.min(axis=None, keepdims=True)) / \
                    #(features_mean.max(axis=None, keepdims=True) -
                    # features_mean.min(axis=None, keepdims=True))

    return features_norm


def regress_measures(predictions, panss):

    r2_score = round(metrics.r2_score(panss, predictions), 2)
    rmse = round(np.sqrt(metrics.mean_squared_error(panss, predictions)), 2)

    print('R2 score: ', r2_score)
    print('RMSE: ', rmse)


def z_score_normalize(data, no_hc):

    data_hc = data[0:no_hc, :]
    data_fes = data[no_hc:, :]

    mean_hc = np.mean(data_hc)
    sd_hc = np.std(data_hc)

    mean_fes = np.mean(data_fes)
    sd_fes = np.std(data_fes)

    mean = np.mean((mean_hc, mean_fes))
    sd = np.mean((sd_hc, sd_fes))

    normalized_data = (data-mean)/sd

    return normalized_data

def get_distances(coeffs, intercept, data):

    norm_normal_vector = np.linalg.norm(coeffs)
    distances = np.apply_along_axis(distance, 1, data, coeffs, intercept, norm_normal_vector)
    return distances


def distance(point, coeffs, intercept, norm_normal_vector):

    distance = np.abs(np.dot(coeffs, point) + intercept) / norm_normal_vector
    return distance


def LOO_svr_distances(features, panss, kernel):

    folds = len(panss)
    predictions = np.zeros((folds,))

    for fold in range(folds):
        features_train = np.delete(features, fold)
        panss_train = np.delete(panss, fold)
        features_train = features_train.reshape(-1, 1)  #single feature

        features_val = features[fold]
        features_val = np.array([[features_val]])

        prediction = regression_model(features_train, panss_train, features_val, kernel)
        predictions[fold] = prediction[0]

    return predictions

