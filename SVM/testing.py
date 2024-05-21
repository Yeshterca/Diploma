# IMPORTS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import nibabel as nib
import numpy as np
import os

dirpath = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\VOXEL-WISE\1st level\ESO_C00221_20150713_1517_1\beta_0001.nii'

betas_subject = nib.load(dirpath).get_fdata()
print(np.shape(betas_subject))

betas_subject_vec = betas_subject.flatten()
print(np.shape(betas_subject_vec))

x = np.isnan(betas_subject)
check = np.where(x, 0, 1)
print(np.sum(check))

#U, V, Vt = np.linalg.svd(betas_centered)

#pc1 = Vt.T[:, 0]

#print(pc1)

pca_VW = PCA(n_components=5)
principal_components = pca_VW.fit_transform(betas_subject_vec)


# def select_components_ICA(principal_components, data):




# LOAD AND PREPARE DATA FROM ICA

#component_numbers, betas_all = get_ica_data(r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\ICA\results\20240124\ICASSO_all2\eso_temporal_regression.mat')
#ic_numbers_selected, ic_betas_selected = select_components(NO_PATIENTS, component_numbers, RELEVANT_ICS, betas_all, NO_COMPONENTS)
#labels = create_labels(NO_HC, NO_FES)  # all above checked

#features_ica = ic_betas_selected.T

# SVM on small sample
#betas_train, labels_train, betas_val, labels_val = split_data(labels, ic_betas_selected.T, NO_PATIENTS)
#labels = labels_train[0:5]
#ic_betas_selected = betas_train[0:5, :]
#ic_betas_selected = prepare_data(ic_betas_selected)
#prediction = svm_classifier(ic_betas_selected, labels, ic_betas_selected)
#print(prediction)
#accuracy, sensitivity, specificity, precision = measures_classifier(labels, prediction)


# SVM
#weights = get_weights(features_ica, labels)
#accuracy, sensitivity, specificity, precision = cross_validation(labels, features_ica, weights, NO_PATIENTS)

#print("ICA")
#print("Accuracy:", accuracy)
#print("Sensitivity:", sensitivity)
#print("Specificity:", specificity)
#print("Precision:", precision)

# plotting scatters
#plt.figure(2)
#plt.scatter(ic_betas_selected[0, :], ic_betas_selected[1, :], c=labels, s=50, cmap='spring')
#plt.show()

# Grid search for gamma
#ic_betas_selected = prepare_data(ic_betas_selected)
#print(np.shape(ic_betas_selected))
#best_gamma = RBF_gamma_search(ic_betas_selected, labels)


# LOAD AND PREPARE DATA FROM VW

#betas_all_vw = load_subjects(r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\VOXEL-WISE\1st level', 'beta_0001.nii', NO_PATIENTS)
#betas_st = prepare_data(betas_all_vw) # normalize

# PCA dim red
#principal_components = pca(betas_st, NO_COMPONENTS)
#features_vw = pca_projection(betas_st, principal_components, NO_PATIENTS, NO_COMPONENTS)

#gamma_VW = RBF_gamma_search(features_vw, labels)

betas_HC_one = np.mean(betas_all[0:NO_HC, :, :, :], axis=0)
betas_FES_one = np.mean(betas_all[NO_HC:, :, :, :], axis=0)
betas_ALL_one = np.mean(betas_all, axis=0)

betas_HC_norm = (betas_HC_one - betas_ALL_one.min(axis=None, keepdims=True)) / \
                            (betas_ALL_one.max(axis=None, keepdims=True) -
                             betas_ALL_one.min(axis=None, keepdims=True))
betas_FES_norm = (betas_FES_one - betas_ALL_one.min(axis=None, keepdims=True)) / \
                            (betas_ALL_one.max(axis=None, keepdims=True) -
                             betas_ALL_one.min(axis=None, keepdims=True))

betas_conc = np.stack((betas_HC_norm, betas_FES_norm), axis=3)


betas_all = load_subjects_3d(r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\VOXEL-WISE\1st level', 'beta_0001.nii', NO_PATIENTS)

betas_conc = mean_groups_for_superpixels(betas_all, NO_HC)

segments, overlay = superpixels(betas_conc, NUM_SEGMENTS, SLICE)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.imshow(overlay[:, :, 0])
ax2.imshow(betas_conc[:, SLICE, :, 0])
ax3.imshow(betas_conc[:, SLICE, :, 1])
plt.show()

sup_betas_all, coords = betas_in_superpixels(betas_conc, segments)

diff = np.zeros((len(sup_betas_all)))
for i in range(len(sup_betas_all)):
    if len(sup_betas_all[i]) == 0:
        diff[i] = 0.
        continue
    mean = np.mean(sup_betas_all[i], axis=1)
    diff[i] = np.abs(mean[0] - mean[1])

sorted_idx = np.flip(np.argsort(diff))  # TODO: check if really descending

betas_all_norm = (betas_all - betas_all.min(axis=None, keepdims=True)) / \
                 (betas_all.max(axis=None, keepdims=True) -
                  betas_all.min(axis=None, keepdims=True))

features_VW = np.zeros((NO_PATIENTS, NO_FEATURES))
for i in range(NO_PATIENTS):
    features_VW[i, :] = get_features(sorted_idx, betas_all_norm[i, :, :, :], coords, NO_FEATURES)


accuracy, sensitivity, specificity, precision = cross_validation(labels, features_VW, weights, NO_PATIENTS)

print("--------------")
print("VOXEL-WISE ANALYSIS")
print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Precision:", precision)

betas_train_large = np.delete(betas_large, fold, 0)
labels_train = np.delete(labels, fold)
weights_train = np.delete(weights, fold)

betas_val = betas_large[fold, :, :, :]
labels_val = labels[fold]

betas_train_small = np.delete(betas_small, fold, 0)
betas_val_small = betas_small[fold, :, :, :]



# accs_vw = []
# no_features = []
# no_segments = []
# no_sup = [200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750]
# for i in range(48, 49):
#     for j in range(0, 1):
#         print('features:', i)
#         NO_FEATURES = i
#         num_seg = NUM_SEGMENTS
#         segments, betas_clustered, coords, overlay1, overlay2, overlay3 = vw_cluster(betas_conc, betas_conc_disp,
#                                                                                      num_seg,
#                                                                                      SLICE)
#         pixel_means, pixel_means_notnorm = vw_get_features_dimred(betas_for_disp, segments, NO_PATIENTS, NO_FEATURES)  #betas_all
#         acc_vw, sen_vw, spec_vw, prec_vw = vw_classify(pixel_means, labels, weights, 'linear', NO_FEATURES)
#         accs_vw.append(acc_vw)
#         no_features.append(i)
#         no_segments.append(j)
#
# plot_features_accuracy(accs_vw, no_features)
#
# display_clustering(overlay1, overlay2, overlay3)
#
# display_mean_groups(betas_conc_disp, SLICE)

#
# with open('accuracies', 'wb') as f:
#     pickle.dump(accs_vw, f)
# with open('no_features', 'wb') as f:
#     pickle.dump(no_features, f)

#betas_large = np.delete(betas_large, [13, 20, 33, 119, 120, 125, 126, 135], axis=0)
#betas_small = np.delete(betas_small, [13, 20, 33, 119, 120, 125, 126, 135], axis=0)
#labels = np.delete(labels, [13, 20, 33, 119, 120, 125, 126, 135])
#weights = get_weights(labels)
