from regression_fc import *
from ica_run import *

# GLOBALS
NO_SUBJECTS = 131
NO_HC = 55
NO_FES = 76
RELEVANT_ICS = np.array([22, 7, 15, 31, 24, 20, 19, 30, 16, 5])
NO_FEATURES_ICA = len(RELEVANT_ICS)
INV_SUBS = np.array([57, 61, 83, 90, 108])
INV_SUBS_VW = np.array([57, 61, 83, 90, 95, 108])
NO_FEATURES_VW = 33
NUM_SEGMENTS = 500


# load PANSS
dir = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\Regression\dataset1_PANSS.xlsx'
panss_all = load_panss(dir)
fes_panss, panss_hc_fes, nan_idx = exclude_inv_fov(panss_all, INV_SUBS, NO_HC)
fes_panss_vw, panss_hc_fes_vw, nan_idx_vw = exclude_inv_fov(panss_all, INV_SUBS_VW, NO_HC)


# load ICA data
directory_ica = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\ICA\results\20240124\ICASSO_fovselect5\eso_temporal_regression.mat'
ica_data_fes, ica_data_all, labels = load_ica_data(directory_ica, NO_SUBJECTS, NO_HC, NO_FES, RELEVANT_ICS, NO_FEATURES_ICA, nan_idx)


# load VW data
directory_vw = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\VOXEL-WISE\1st_level_fovselected5'
vw_data_large, vw_data_small = load_VW_data(directory_vw, 'beta_0001.nii', NO_SUBJECTS-1, NO_HC, nan_idx_vw)


# regression ICA
predictions_ica = LOO_svr_ica(ica_data_fes, fes_panss[:, 3], 'poly')
#predictions_ica = regression_model(ica_data_fes, fes_panss[:, 6], ica_data_fes, 'poly')
#print(np.min(panss[:, 3]))
#print(np.max(panss[:, 3]))
print('------- ICA regression -------')
regress_measures(predictions_ica, fes_panss[:, 3])


# regression VW
#predictions_vw = LOO_svr_vw(vw_data_large, vw_data_small, fes_panss_vw[:, 0], NUM_SEGMENTS, NO_FEATURES_VW, 'poly')
#print('------- VW regression -------')
#regress_measures(predictions_vw, fes_panss_vw[:, 0])


predictions_hc = np.zeros((NO_HC,))
deleted_fes = np.where(nan_idx)[0]
predictions_fes = np.insert(predictions_ica, deleted_fes, 0)
predictions_hc_fes = np.concatenate((predictions_hc, predictions_fes), axis=0)
#
# # display results
pca = PCA(n_components=2)
ica_pca_all = pca.fit_transform(ica_data_all)
ica_pca_fes = pca.fit_transform(ica_data_fes)

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
# plt.scatter(ica_pca_all[:, 0], ica_pca_all[:, 1], c=labels, cmap='viridis')
# plt.colorbar(label='PANSS')
# plt.title('Original data - HC & FES')

plt.subplot(1, 2, 1)
plt.scatter(ica_pca_all[:, 0], ica_pca_all[:, 1], c=panss_hc_fes[:, 3], cmap='viridis')  # c=panss[:, 3], ica_pca_fes
plt.colorbar(label='PANSS')
plt.title('Original data')

plt.subplot(1, 2, 2)
plt.scatter(ica_pca_all[:, 0], ica_pca_all[:, 1], c=predictions_hc_fes, cmap='viridis')  # predictions_ica
plt.colorbar(label='Predictions')
plt.title('SVR predictions (of FES), ICA')
#
# plt.subplot(1, 3, 3)
# plt.scatter(ica_pca[:, 0], ica_pca[:, 1], c=predictions_vw, cmap='viridis')
# plt.colorbar(label='Predictions')
# plt.title('SVR predictions, VW')
#

plt.tight_layout
plt.show()

coeffs = np.load(r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\Regression\coeffs.npy')
intercept = np.load(r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\Regression\intercept.npy')

distances = get_distances(coeffs, intercept, ica_data_fes)

predictions_distances_ica = LOO_svr_distances(distances, fes_panss[:, 3], 'poly')
print('------- ICA regression -------')
regress_measures(predictions_distances_ica, fes_panss[:, 3])


