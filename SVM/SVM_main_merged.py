# IMPORTS
from ica_run import *
from nudztest_fc import *
from vw_run import *
from features_optim import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nilearn import plotting
import numpy as np
import pickle

# GLOBALS
NO_FEATURES_VW = 17
NUM_SEGMENTS = 500
SLICE = 55

SUBJECTS_IKEM = 131
SUBJECTS_NUDZ = 158
HC_IKEM = 55
FES_IKEM = 76
HC_NUDZ = 66
FES_NUDZ = 92

RELEVANT_ICS = np.array([34, 18, 22, 20, 8, 6, 5, 17, 21, 26, 31])
NO_FEATURES_ICA = len(RELEVANT_ICS)

directory_ica = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\data_nudz\ICA\merged_ikem_nudz\eso_temporal_regression.mat'
directory_vw_ikem = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\VOXEL-WISE\1st_level_fovselected5'
directory_vw_nudz = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\data_nudz\NUDZ_vw\1st_level'

### ICA

# get data
ica_features_ikem, labels_ikem, weights_ikem, ica_features_nudz, labels_nudz, weights_nudz = ica_load_2datasets(directory_ica, SUBJECTS_IKEM, HC_IKEM, FES_IKEM, SUBJECTS_NUDZ, HC_NUDZ, FES_NUDZ, RELEVANT_ICS, NO_FEATURES_ICA)

# random shuffle
r = np.array(range(labels_ikem.shape[0]))
np.random.shuffle(r)
labels_s_ikem = labels_ikem[r]
weights_s_ikem = weights_ikem[r]
ica_features_s_ikem = ica_features_ikem[r, :]

r = np.array(range(labels_nudz.shape[0]))
np.random.shuffle(r)
labels_s_nudz = labels_nudz[r]
weights_s_nudz = weights_nudz[r]
ica_features_s_nudz = ica_features_nudz[r, :]

# classification
print('------- ICA -------')
accuracy, sensitivity, specificity, precision = classify_ica_nudz(ica_features_s_ikem, labels_s_ikem, weights_s_ikem, ica_features_s_nudz, labels_s_nudz, weights_s_nudz, 'linear')


### VW
betas_large_ikem, betas_small_ikem, labels_ikem, weights_ikem = vw_load_data(directory_vw_ikem, 'beta_0001.nii', SUBJECTS_IKEM-1, HC_IKEM, FES_IKEM-1)
betas_large_nudz, betas_small_nudz, labels_nudz, weights_nudz = vw_load_data(directory_vw_nudz, 'beta_0001.nii', SUBJECTS_NUDZ, HC_NUDZ, FES_NUDZ)

#betas_small_ikem, betas_small_nudz = z_score_normalize2(betas_small_ikem, betas_small_nudz, HC_IKEM, HC_NUDZ)

# print('min ikem: ', np.min(betas_small_ikem))
# print('min nudz: ', np.min(betas_large_nudz))
# print('max ikem: ', np.max(betas_small_ikem))
# print('max nudz: ', np.max(betas_small_nudz))
# print('mean ikem: ', np.mean(betas_small_ikem))
# print('mean nudz: ', np.mean(betas_small_nudz))
# print('std ikem: ', np.std(betas_small_ikem))
# print('std nudz: ', np.std(betas_small_nudz))

segments, overlay1, overlay2, overlay3, betas_conc_small = cluster(betas_large_ikem, betas_small_ikem, HC_IKEM, NUM_SEGMENTS, SLICE)
pls_ikem, pls, pixel_means_ikem = vw_features_find(betas_small_ikem, labels_ikem, segments, NO_FEATURES_VW, SUBJECTS_IKEM)

pixel_means_nudz, pixel_means_nudz_nn = pixel_means(betas_small_nudz, segments, SUBJECTS_NUDZ)
pls_nudz = pls.transform(pixel_means_nudz.T)

predictions, probs = svm_classifier(pls_ikem, labels_ikem, weights_ikem, pls_nudz, kernel='linear')


# # np.save('predictions_vw_nudz_test.npy', predictions)
# # np.save('probabilities_vw_nudz_test.npy', probs)
# #
# # print(predictions)
# # print(labels_nudz)
# #
print('------- VW -------')
accuracy, sensitivity, specificity, precision = get_measures(labels_nudz, predictions, weights_nudz)
# #
# # print(labels_nudz)
# # print(predictions)
# #
fpr, tpr, _ = roc_curve(labels_nudz, probs)
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