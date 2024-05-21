import numpy as np
from get_data_ICA import *
from model import *
from sklearn.metrics import roc_curve, auc


# LOAD AND PREPARE DATA FROM ICA
def ica_load_prep_data(directory, no_subjects, no_hc, no_fes, relevant_ics, no_components):

    # load all
    component_numbers, betas_all = get_ica_data(directory)
    # select significant ics
    ic_numbers_selected, ic_betas_selected = select_components(no_subjects, component_numbers, relevant_ics, betas_all, no_components)

    # get labels and weights
    labels = create_labels(no_hc, no_fes)
    weights = get_weights(labels)

    # norm
    #betas = (2 * (ic_betas_selected - ic_betas_selected.min(axis=None, keepdims=True)) / (ic_betas_selected.max(axis=None, keepdims=True) - ic_betas_selected.min(axis=None, keepdims=True))) - 1
    betas = z_score_normalize(ic_betas_selected.T, no_hc)
    features_ica = betas #z_score_normalize(ic_betas_selected.T, no_hc)

    return features_ica, labels, weights


# CLASSIFY
def ica_classify(betas, labels, weights, kernel):

    predictions, probs = ica_LOO_cross_validation(betas, labels, weights, kernel)  #, probs

    #np.save('predictions_ica_nudz.npy', predictions)
    #np.save('probs_ica_nudz.npy', probs)

    print('---------- ICA-CLASSIFICATION ----------')

    accuracy, sensitivity, specificity, precision = get_measures(labels, predictions, weights)

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ICA: Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


    return accuracy, sensitivity, specificity, precision

