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

no_subjects_ikem, no_hc_ikem, no_fes_ikem, relevant_ics_ikem, no_features_ica_ikem, directory_ica_ikem, directory_vw_ikem = set_vars('ikem')
no_subjects_nudz, no_hc_nudz, no_fes_nudz, relevant_ics_nudz, no_features_ica_nudz, directory_ica_nudz, directory_vw_nudz = set_vars('nudz')

### ICA

ica_features_ikem, labels_ikem, weights_ikem = ica_load_prep_data(directory_ica_ikem, no_subjects_ikem, no_hc_ikem, no_fes_ikem, relevant_ics_ikem, no_features_ica_ikem)
ica_features_nudz, labels_nudz, weights_nudz = ica_load_prep_data(directory_ica_nudz, no_subjects_nudz, no_hc_nudz, no_fes_nudz, relevant_ics_nudz, no_features_ica_nudz)

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

print('------- ICA -------')
accuracy, sensitivity, specificity, precision = classify_ica_nudz(ica_features_s_ikem, labels_s_ikem, weights_s_ikem, ica_features_s_nudz, labels_s_nudz, weights_s_nudz, 'linear')

### VW

betas_large_ikem, betas_small_ikem, labels_ikem, weights_ikem = vw_load_data(directory_vw_ikem, 'beta_0001.nii', no_subjects_ikem-1, no_hc_ikem, no_fes_ikem-1)
betas_large_nudz, betas_small_nudz, labels_nudz, weights_nudz = vw_load_data(directory_vw_nudz, 'beta_0001.nii', no_subjects_nudz, no_hc_nudz, no_fes_nudz)

segments, overlay1, overlay2, overlay3, betas_conc_small = cluster(betas_large_ikem, betas_small_ikem, no_hc_ikem, NUM_SEGMENTS, SLICE)
pls_ikem, pls, pixel_means_ikem = vw_features_find(betas_small_ikem, labels_ikem, segments, NO_FEATURES_VW, no_subjects_ikem)

pixel_means_nudz, pixel_means_nudz_nn = pixel_means(betas_small_nudz, segments, no_subjects_nudz)
pls_nudz = pls.transform(pixel_means_nudz.T)

predictions, probs = svm_classifier(pls_ikem, labels_ikem, weights_ikem, pls_nudz, kernel='linear')


# np.save('predictions_vw_nudz_test.npy', predictions)
# np.save('probabilities_vw_nudz_test.npy', probs)
#
# print(predictions)
# print(labels_nudz)
#
print('------- VW -------')
accuracy, sensitivity, specificity, precision = get_measures(labels_nudz, predictions, weights_nudz)
#
# print(labels_nudz)
# print(predictions)
#
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

