from model import *
from vw_clustering import *
from plots_figures import *


def vw_load_data(directory, filename, no_subjects, no_hc, no_fes):
    """
    Loads and prepares data, labels and weights.

    :param directory: full path to data directory
    :param filename: name of file with desired data (betas_001.nii)
    :param no_subjects: number of subjects
    :param no_hc: number of healthy
    :param no_fes: number of patients
    :return: betas_large ... masked data with large background for clustering (10)
             betas_small ... masked data with small backgroud (0)
             [no_subjects, 91, 109, 91]
    """

    betas_large, betas_small = load_subjects_3d(directory, filename, no_subjects, no_hc)
    labels = create_labels(no_hc, no_fes)
    weights = get_weights(labels)

    return betas_large, betas_small, labels, weights

def vw_LOO_cluster_classify(betas_large, betas_small, labels, weights, no_subjects, no_segments, no_slice, no_features):
    """
    Leave-one-out loop performs split to training and validation data, clustering, PLS projection and classification.

    :param betas_large: masked data with large background for clustering (10), [no_subjects, 91, 109, 91]
    :param betas_small: asked data with small backgroud (0), [no_subjects, 91, 109, 91]
    :param labels: (no_subjects, ), 0/1
    :param weights: (no_subjects, ), weights to compensate for uneven dataset
    :param no_subjects: number of subjects
    :param no_segments: number of superpixels to find
    :param no_slice: number of slice to display
    :param no_features: number of features for classification
    :return: predictions ... (no_subjects, ), 0/1
             probs ... (no_subjects, ), probabilities from SVM
    """

    folds = len(labels)
    predictions = np.zeros((folds,))
    probs = np.zeros((folds,))

    for fold in range(folds):
        print('FOLD: ', fold)
        if fold <= 55:
            no_hc = 54
        else:
            no_hc = 55

        betas_train_large, betas_train_small, labels_train, weights_train, betas_val = data_split(betas_large,
                                                                                                  betas_small, labels,
                                                                                                  weights, fold)

        segments, overlay1, overlay2, overlay3, betas_conc_small = cluster(betas_train_large, betas_train_small, no_hc, no_segments, 53)

        pls_train, pls, pixel_means_all = vw_features_find(betas_train_small, labels_train, segments, no_features, no_subjects)
        pls_val = vw_features_transform(pls, betas_val, segments, pixel_means_all)

        prediction, prob = svm_classifier(pls_train, labels_train, weights_train, pls_val, kernel='linear')
        predictions[fold] = prediction[0]
        probs[fold] = prob[0]

    return predictions, probs, betas_conc_small, overlay1, overlay2, overlay3


def data_split(betas_large, betas_small, labels, weights, fold):
    """
    Split data to training and validation.

    :param betas_large: masked data with large background for clustering (10), [no_subjects, 91, 109, 91]
    :param betas_small: asked data with small backgroud (0), [no_subjects, 91, 109, 91]
    :param labels: (no_subjects, ), 0/1
    :param weights: (no_subjects, ), weights to compensate for uneven dataset
    :param fold: "fold number"
    :return: betas_train_large ... training data, large background [no_subjects-1, 91, 109, 91]
             betas_train_small ... training data, small background [no_subjects-1, 91, 109, 91]
             labels_train ... training labels (no_subjects-1, )
             weights_train ... training weights (no_subjects-1, )
             betas_val ... validation data [1, 91, 109, 91]
    """

    betas_train_large = np.delete(betas_large, fold, 0)  # TODO axis
    betas_train_small = np.delete(betas_small, fold, 0)
    labels_train = np.delete(labels, fold)
    weights_train = np.delete(weights, fold)

    #betas_val = betas_small[fold, :, :, :]  # normally in LOO ok
    betas_val = betas_small[fold, :]  # in case of features opt.

    return betas_train_large, betas_train_small, labels_train, weights_train, betas_val


def cluster(betas_train_large, betas_train_small, no_hc, no_segments, no_slice):
    """
    Cluster data using SLIC superpixels.

    :param betas_train_large: training data, large background [no_subjects-1, 91, 109, 91]
    :param betas_train_small: training data, small background [no_subjects-1, 91, 109, 91]
    :param no_hc: number of healthy
    :param no_segments: number of superpixels for clustering
    :param no_slice: number of slice to display
    :return: segments ... clustered data (labels)
             overlay1 ... slice to display clustering, transversal
             overlay2 ... frontal
             overlay3 ... sagital
    """

    betas_conc_large = mean_groups_for_superpixels(betas_train_large, no_hc)
    betas_conc_small = mean_groups_for_superpixels(betas_train_small, no_hc)
    segments, overlay1, overlay2, overlay3 = superpixels(betas_conc_large, betas_conc_small, no_segments, no_slice)

    return segments, overlay1, overlay2, overlay3, betas_conc_small


def vw_features_find(betas_train_small, labels_train, segments, no_features, no_subjects):
    """
    Find PLS transformation and transform training data.

    :param betas_train_small: training data, small background [no_subjects-1, 91, 109, 91]
    :param labels_train: training labels (no_subjects-1, )
    :param segments: clustered data (labels)
    :param no_features: number of features for SVM
    :param no_subjects: number of subjects
    :return: pls_train ... pls tranformation fitted to training data
             pls ... pls transformation learned on training data
             pixel_means_all ... means of superpixels
    """

    pixel_means_train, pixel_means_all = pixel_means(betas_train_small, segments, no_subjects-1)  # -1 ... in LOO
    pls_train, pls = get_features_pls(pixel_means_train.T, labels_train, no_features)  # .T pixel means

    return pls_train, pls, pixel_means_all


def vw_features_transform(pls, betas_val, segments, pixel_means_all):
    """
    Transform validation data using PLS transformation found on training data.

    :param pls: transformation
    :param betas_val: validation data [1, 91, 109, 91]
    :param segments: clustered data (labels)
    :param pixel_means_all: means of superpixels
    :return: pls_val ... transformed validation data
    """

    clusted_subject = betas_in_superpixels(betas_val, segments)
    pixel_means_val = mean_superpixels(clusted_subject)
    #pixel_means_val = -1 + 2 * (pixel_means_val - np.min(pixel_means_all)) / (np.max(pixel_means_all) - np.min(pixel_means_all))

    pixel_means_val = pixel_means_val.reshape(1, -1)
    pls_val = pls.transform(pixel_means_val)

    return pls_val











