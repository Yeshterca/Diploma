# IMPORTS
from ica_run import *
from vw_run import *
from features_optim import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nilearn import plotting
import numpy as np
import pickle
import scipy.io

dataset = 'ikem'

if dataset == 'ikem':
    # GLOBALS IKEM
    NO_SUBJECTS = 130  #131  #136 #119
    NO_HC = 55  #55  #49
    NO_FES = 75  #76  #81  #70
    RELEVANT_ICS = np.array([22, 7, 15, 31, 24, 20, 19, 30, 16, 5])  # check 16, 5  #np.array([22, 7, 15, 31, 24])  #np.array([12, 23, 9, 29, 7])
    NO_FEATURES_ICA = len(RELEVANT_ICS)
    directory_ica = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\ICA\results\20240124\ICASSO_fovselect5\eso_temporal_regression.mat'
    directory_vw = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\VOXEL-WISE\1st_level_fovselected5'

elif dataset == 'nudz':
    # GLOBALS NUDZ
    NO_SUBJECTS = 158
    NO_HC = 66
    NO_FES = 92
    RELEVANT_ICS = np.array([18, 28, 22, 21, 10, 17, 19, 20, 12, 34])  # run 1: np.array([17, 28, 29, 10, 19, 20, 12, 33])
    NO_FEATURES_ICA = len(RELEVANT_ICS)
    directory_ica = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\data_nudz\ICA\results2\eso_temporal_regression.mat'
    directory_vw = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\data_nudz\NUDZ_vw\1st_level'


NO_FEATURES_VW = 34
NUM_SEGMENTS = 500
SLICE = 55

### FIND OPT. NUMBER OF FEATURES
#directory_vw = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\VOXEL-WISE\1st_level_fovselected5'
#betas_large, betas_small, labels, weights = vw_load_data(directory_vw, 'beta_0001.nii', NO_SUBJECTS, NO_HC, NO_FES)
#segments, _, _, _, _ = cluster(betas_large, betas_small, NO_HC, NUM_SEGMENTS, SLICE)
#optim_features(betas_small, labels, weights, segments, NO_SUBJECTS)

### ICA
#ica_features, labels, weights = ica_load_prep_data(directory_ica, NO_SUBJECTS, NO_HC, NO_FES, RELEVANT_ICS, NO_FEATURES_ICA)
#acc_ica, sen_ica, spec_ica, prec_ica = ica_classify(ica_features, labels, weights, 'linear')

### VW
features = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
no_f = len(features)
acc = np.zeros((no_f, 1))
sens = np.zeros((no_f, 1))
spec = np.zeros((no_f, 1))
aucs = np.zeros((no_f, 1))

for i in range(no_f):
    NO_FEATURES_VW = features[i]
    betas_large, betas_small, labels, weights = vw_load_data(directory_vw, 'beta_0001.nii', NO_SUBJECTS, NO_HC, NO_FES)
    predictions, probs, betas_conc_small, overlay1, overlay2, overlay3 = vw_LOO_cluster_classify(betas_large, betas_small, labels, weights, NO_SUBJECTS, NUM_SEGMENTS, SLICE, NO_FEATURES_VW)
    accuracy, sensitivity, specificity, precision = get_measures(labels, predictions, weights)
    acc[i] = accuracy
    sens[i] = sensitivity
    spec[i] = specificity
    print(acc)

    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('VW: Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    aucs[i] = roc_auc

np.save('accs_opt.npy', acc)
np.save('sens_opt.npy', sens)
np.save('spec_opt.npy', spec)
np.save('aucs_opt.npy', aucs)

#np.save('predictions_vw_33f.npy', predictions)
#np.save('probabilities_vw_33f.npy', probs)


### MOTION CHECK
# predictions = np.load(r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\SVM\results\NUDZ\predictions_vw_nudz.npy')
# probs = np.load(r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\SVM\results\NUDZ\probabilities_vw_nudz.npy')
# labels = create_labels(NO_HC, NO_FES)

# def access_elements(nums, list_index):
#     result = [nums[i] for i in list_index]
#     return result
#
# motion_out_pred = access_elements(predictions, [13, 20, 33, 119, 120, 125, 126, 135])
# motion_out_label = access_elements(labels, [13, 20, 33, 119, 120, 125, 126, 135])
#
# print(motion_out_pred)
# print(motion_out_label)

### ROC
# fpr, tpr, _ = roc_curve(labels, probs)
# roc_auc = auc(fpr, tpr)
#
# plt.figure()
# plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('VW: Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.show()

### PLOTS
#rois_out = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\preprocess\rois_out_ikem.npy'
#plot_rois_distribution(rois_out)

# display_clustering(overlay1, overlay2, overlay3)
# #
# display_mean_groups(betas_conc_small, SLICE)



# # reset
# ica_features = deepcopy(ica_features_)
# principal_components = deepcopy(principal_components_)
# labels = deepcopy(labels_)
# weights = deepcopy(weights_)


# # fig, (ax1, ax2, ax3, ax4, ax7, ax8) = plt.subplots(1, 6, figsize=(20, 10))
# # #ax1.imshow(betas_conc_disp[:, :, SLICE, 0])
# # #ax1.set_title('HC group mean')
# # #ax2.imshow(betas_conc_disp[:, :, SLICE, 1])
# # #ax2.set_title('FES group mean')
# # ax1.imshow(best_sups[:, SLICE, :])
# # ax2.imshow(overlay3[:, :, 0])
# # ax3.imshow(best_sups[:, :, SLICE])
# # #ax3.set_title('best sup. = features')
# # ax4.imshow(overlay1[:, :, 0])
# # #ax4.set_title('superpixels')
# #
# # #ax5.imshow(betas_conc_disp[SLICE, :, :, 0])
# # #ax6.imshow(betas_conc_disp[SLICE, :, :, 1])
# # ax7.imshow(best_sups[SLICE, :, :])
# # ax8.imshow(overlay2[:, :, 0])
# #
# # fig.suptitle('SLICE 60', fontsize=30)
# # plt.show()

