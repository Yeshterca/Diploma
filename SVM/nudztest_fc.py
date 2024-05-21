from model import *
from sklearn.metrics import roc_curve, auc
from get_data_ICA import *

def set_vars(dataset):
    if dataset == 'ikem':
        no_subjects = 131
        no_hc = 55
        no_fes = 76
        relevant_ics = np.array([22, 7, 15, 31, 24, 20, 19, 30])
        no_features_ica = len(relevant_ics)
        directory_ica = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\ICA\results\20240124\ICASSO_fovselect5\eso_temporal_regression.mat'
        directory_vw = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\VOXEL-WISE\1st_level_fovselected5'

    elif dataset == 'nudz':
        no_subjects = 158
        no_hc = 66
        no_fes = 92
        relevant_ics = np.array([17, 28, 29, 10, 19, 20, 12, 33])
        no_features_ica = len(relevant_ics)
        directory_ica = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\data_nudz\ICA\results2\eso_temporal_regression.mat'
        directory_vw = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\data_nudz\NUDZ_vw\1st_level'

    return no_subjects, no_hc, no_fes, relevant_ics, no_features_ica, directory_ica, directory_vw


def classify_ica_nudz(ica_features_ikem, labels_ikem, weights_ikem, ica_features_nudz, labels_nudz, weights_nudz, kernel):

    predictions, probs = svm_classifier(ica_features_ikem, labels_ikem, weights_ikem, ica_features_nudz, kernel)

    accuracy, sensitivity, specificity, precision = get_measures(labels_nudz, predictions, weights_nudz)

    fpr, tpr, _ = roc_curve(labels_nudz, probs)
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


def ica_load_2datasets(directory, subjects_ikem, hc_ikem, fes_ikem, subjects_nudz, hc_nudz, fes_nudz, relevant_ics, no_components):

    # load all
    component_numbers, betas_all = get_ica_data(directory)
    # select significant ics
    ic_numbers_selected, ic_betas_selected = select_components(subjects_ikem+subjects_nudz, component_numbers, relevant_ics, betas_all, no_components)

    # separate to IKEM / NUDZ
    betas_ikem = ic_betas_selected[:, 0:subjects_ikem].T
    betas_nudz = ic_betas_selected[:, subjects_ikem:].T

    # get labels and weights
    labels_ikem = create_labels(hc_ikem, fes_ikem)
    weights_ikem = get_weights(labels_ikem)
    labels_nudz = create_labels(hc_nudz, fes_nudz)
    weights_nudz = get_weights(labels_nudz)

    # normalize
    features_ikem = z_score_normalize(betas_ikem, hc_ikem)
    features_nudz = z_score_normalize(betas_nudz, hc_nudz)

    return features_ikem, labels_ikem, weights_ikem, features_nudz, labels_nudz, weights_nudz


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


def z_score_normalize2(ikem, nudz, hc_ikem, hc_nudz):

    data_hc = ikem[0:hc_ikem, :]
    data_fes = ikem[hc_ikem:, :]

    mean_hc = np.mean(data_hc)
    sd_hc = np.std(data_hc)

    mean_fes = np.mean(data_fes)
    sd_fes = np.std(data_fes)

    mean_ikem = np.mean((mean_hc, mean_fes))
    sd_ikem = np.mean((sd_hc, sd_fes))

    data_hc = nudz[0:hc_nudz, :]
    data_fes = nudz[hc_nudz:, :]

    mean_hc = np.mean(data_hc)
    sd_hc = np.std(data_hc)

    mean_fes = np.mean(data_fes)
    sd_fes = np.std(data_fes)

    mean_nudz = np.mean((mean_hc, mean_fes))
    sd_nudz = np.std((sd_hc, sd_fes))

    mean = np.mean((mean_ikem, mean_nudz))
    sd = np.std((sd_ikem, sd_nudz))

    ikem = (ikem - mean)/sd
    nudz = (nudz - mean)/sd

    return ikem, nudz


