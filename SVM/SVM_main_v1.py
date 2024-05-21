# IMPORTS
from ica_run import *
from vw_run import *
from plots_figures import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nilearn import plotting
import numpy as np
import pickle


# GLOBALS
NO_PATIENTS = 131 #131  #136 #119
NO_HC = 55  #55  #49
NO_FES = 76  #76  #81  #70
RELEVANT_ICS = np.array([22, 7, 15, 31, 24, 20, 19, 30])  #np.array([22, 7, 15, 31, 24])  #np.array([12, 23, 9, 29, 7])
NO_COMPONENTS = len(RELEVANT_ICS)
NO_FEATURES = 33
NUM_SEGMENTS = 500
SLICE = 55

### ICA
directory_ica = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\ICA\results\20240124\ICASSO_fovselect5\eso_temporal_regression.mat'
ica_features, labels, weights = ica_load_prep_data(directory_ica, NO_PATIENTS, NO_HC, NO_FES, RELEVANT_ICS, NO_COMPONENTS, 'classify')

# r = np.array(range(labels.shape[0]))
# np.random.shuffle(r)
# labels = labels[r]
# ica_features = ica_features[r, :]
# weights = weights[r]

acc_ica, sen_ica, spec_ica, prec_ica = ica_classify(ica_features, labels, weights, 'linear')


### VW
# NO_PATIENTS = NO_PATIENTS  # TODO !!!!
directory_vw = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\VOXEL-WISE\1st_level_fovselected5'
# betas_all, betas_conc, betas_conc_disp, betas_for_disp = vw_load_prep_data(directory_vw, NO_PATIENTS, NO_HC, 130)
# labels = create_labels(NO_HC, NO_FES)
# weights = get_weights(labels)
# pls_train = 0
# pls = 0
#
# # all
# segments, betas_clustered, coords, overlay1, overlay2, overlay3 = vw_cluster(betas_conc, betas_conc_disp, NUM_SEGMENTS, SLICE)
# pixel_means, pixel_means_notnorm = vw_get_features_dimred(betas_for_disp, segments, NO_PATIENTS, NO_FEATURES)  #betas_all
# acc_vw, sen_vw, spec_vw, prec_vw, pls = vw_classify(pixel_means, labels, weights, 'linear', NO_FEATURES, pls, 0)
#
# display_clustering(overlay1, overlay2, overlay3)
# display_mean_groups(betas_conc_disp, SLICE)
#
# # superpixels on -1
# betas_all2, betas_conc2, betas_conc_disp2, betas_for_disp2 = vw_load_prep_data(directory_vw, NO_PATIENTS, NO_HC, 129)
# labels2 = create_labels(NO_HC, NO_FES-1)
# weights2 = get_weights(labels2)
# segments2, _, _, o1, o2, o3 = vw_cluster(betas_conc2, betas_conc_disp2, NUM_SEGMENTS, SLICE)
# pixel_means, pixel_means_notnorm = vw_get_features_dimred(betas_for_disp2, segments2, NO_PATIENTS-1, NO_FEATURES)
# acc_vw, sen_vw, spec_vw, prec_vw, pls = vw_classify(pixel_means[:522, :], labels2, weights2, 'linear', NO_FEATURES, pls, 1)
#
# display_clustering(o1, o2, o3)
# display_mean_groups(betas_conc_disp2, SLICE)



### VW

betas_large, betas_small = load_subjects_3d(directory_vw, 'beta_0001.nii', NO_PATIENTS, 130)
labels = create_labels(NO_HC, NO_FES)
weights = get_weights(labels)

folds = len(labels)
predictions = np.zeros((folds,))
probs = np.zeros((folds,))

no_hc = 0
for fold in range(folds):
    print('FOLD:', fold)

    if fold <= 55:
        no_hc = 54
    else:
        no_hc = 55

    # train and validation
    betas_train_large = np.delete(betas_large, fold, 0)
    betas_train_small = np.delete(betas_small, fold, 0)
    labels_train = np.delete(labels, fold)
    weights_train = np.delete(weights, fold)

    betas_val = betas_small[fold, :, :, :]
    labels_val = labels[fold]

    # # masking nans
    # mask = np.ones_like(betas_train[0, :, :, :])
    # count_nans = np.zeros_like(betas_all[0, :, :, :])
    #
    # for j in range(0, NO_PATIENTS - 1):   # TODO: 1 ... val, 1 ... sub92
    #     nan_mask = np.isnan(betas_train[j, :, :, :])
    #     count_nans = count_nans + nan_mask
    # mask[count_nans > 0] = 0
    #
    # betas_nan0 = betas_train
    # betas_nanout = np.isnan(betas_train)
    # betas_nan0[betas_nanout] = 0
    # betas_train_small = np.zeros_like(betas_train)
    #
    # for j in range(0, NO_PATIENTS - 1):  # TODO
    #     betas_train_small[j, :, :, :] = betas_nan0[j, :, :, :] * mask
    #
    # betas_train_large = np.where(betas_train_small == 0., 10., betas_train_small)
    #
    # betas_nanout = np.isnan(betas_val)
    # betas_val[betas_nanout] = 0
    # betas_val = betas_val * mask

    # concatenate and cluster training data
    betas_conc_large = mean_groups_for_superpixels(betas_train_large, no_hc)
    betas_conc_small = mean_groups_for_superpixels(betas_train_small, no_hc)

    segments_train, betas_clustered_train, coords_train, overlay1_train, overlay2_train, overlay3_train = vw_cluster(betas_conc_large, betas_conc_small,
                                                                                 NUM_SEGMENTS, SLICE)

    #display_clustering(overlay1_train, overlay2_train, overlay3_train)
    #display_mean_groups(betas_conc_small, SLICE)

    # cluster validation data
    #segments_val, betas_clustered_val, coords_val, overlay1_val, overlay2_val, overlay3_val = vw_cluster(betas_val, betas_val_disp, NUM_SEGMENTS, SLICE)

    # get features on training data
    pixel_means_train, pixel_means_all = vw_get_features_dimred(betas_train_small, segments_train, NO_PATIENTS-1)
    pls_train, pls = get_features_pls(pixel_means_train.T, labels_train, NO_FEATURES)

    # get features on validation data
    betas_valsubj_clustered = betas_in_superpixels(betas_val, segments_train)
    pixel_means_val = mean_superpixels(betas_valsubj_clustered)
    pixel_means_val = -1 + 2 * (pixel_means_val - np.min(pixel_means_all)) / (np.max(pixel_means_all) - np.min(pixel_means_all))

    pixel_means_val = pixel_means_val.reshape(1, -1)
    pls_val = pls.transform(pixel_means_val)

    # predict
    prediction, prob = svm_classifier(pls_train, labels_train, weights_train, pls_val, kernel='linear')
    #print('pred2: ', prediction)
    #print('prob2: ', prob)

    predictions[fold] = prediction[0]
    probs[fold] = prob[0]


# get measures
accuracy, sensitivity, specificity, precision = get_measures(labels, predictions, weights)


# ROC

fpr, tpr, _ = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('VW: Receiver Operating Characteristic (ROC) Curve, 33f')
plt.legend(loc="lower right")
plt.show()


### PLOTS
rois_out = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\SVM\rois_out.npy'
plot_rois_distribution(rois_out)

display_clustering(overlay1_train, overlay2_train, overlay3_train)

display_mean_groups(betas_conc_small, SLICE)



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



#
# ### ICA
# directory_ica = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\ICA\results\20240124\ICASSO_fovselect5\eso_temporal_regression.mat'
#
# ica_features, labels, weights = ica_load_prep_data(directory_ica, NO_PATIENTS, NO_HC, NO_FES, RELEVANT_ICS, NO_COMPONENTS)
#
# #acc_ica, sen_ica, spec_ica, prec_ica = ica_classify(ica_features, labels, weights, NO_PATIENTS)
# acc_ica, sen_ica, spec_ica, prec_ica = ica_classify(ica_features, labels, weights, 'linear')
#
# #acc2 = LOO_cross(ica_features, labels, weights)
#
#
# ### VOXEL-WISE ANALYSIS
# directory_vw = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\VOXEL-WISE\1st_level_fovselected5'
#
# betas_all, betas_conc, betas_conc_disp = vw_load_prep_data(directory_vw, NO_PATIENTS, NO_HC)
# segments, betas_clustered, coords, overlay1, overlay2, overlay3 = vw_cluster(betas_conc, betas_conc_disp, NUM_SEGMENTS, SLICE)
# #vw_features, best_sups = vw_get_features(betas_clustered, betas_all, coords, NO_PATIENTS, NO_FEATURES)
#
# #acc_vw, sen_vw, spec_vw, prec_vw = vw_classify(vw_features, labels, weights, NO_PATIENTS)
#
#
# #new_image = nib.Nifti1Image(best_sups, affine=np.eye(4))
# #nib.save(new_image, r"C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\SVM\best_sup_v2")
#
#

#
#
# principal_components = vw_get_features_pca(betas_all, segments, NO_PATIENTS, NO_FEATURES)
#
# acc_vw, sen_vw, spec_vw, prec_vw = ica_classify(principal_components, labels, weights, 'linear')
#
