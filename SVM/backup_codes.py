import h5py
import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# GLOBALS
NO_PATIENTS = 136
NO_HC = 55
NO_FES = 81
RELEVANT_ICS = np.array([12, 23, 9, 29, 7])

# LOAD AND PREPARE DATA

# Load temporal regression results

tempreg = h5py.File(r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\ICA\results\20240124\ICASSO_all2\eso_temporal_regression.mat', 'r')

components_table = tempreg['regressInfo/componentNumbers']
components_array = np.squeeze(np.array(components_table))
components_numbers = components_array.astype('int')

betas_table = tempreg['regressInfo/regressionParameters']
betas_array = np.array(betas_table)
betas_all = betas_array.astype('float64')


# Keep selected components
IC_numbers_selected = []
IC_betas_selected = np.zeros((5, NO_PATIENTS))
for i in range(0, 5):
    idx = np.where(components_numbers == RELEVANT_ICS[i])[0][0]
    IC_numbers_selected.append(idx)
    IC_betas_selected[i, :] = betas_all[idx, :]


# Separate patients
betas_per_subject = []
for i in range(0, NO_PATIENTS):
    betas_subject = IC_betas_selected[:, i]
    betas_per_subject.append(betas_subject)

# Create labels
HC_labels = np.zeros((1, NO_HC), 'int')
FES_labels = np.ones((1, NO_FES), 'int')
labels = np.squeeze(np.concatenate((HC_labels, FES_labels), axis=1))


# Shuffle data
r = np.array(range(labels.shape[0]))
np.random.shuffle(r)
labels_shuffled = labels[r]
betas_shuffled = IC_betas_selected.T[r, :]


# Split to training and validation sets
split_idx = int(NO_PATIENTS * 0.8)

betas_train = betas_shuffled[:split_idx, :]
labels_train = labels_shuffled[:split_idx]

betas_val = betas_shuffled[split_idx+1:, :]
labels_val = labels_shuffled[split_idx+1:]

#print(np.shape(betas_train))
#print(np.shape(betas_val))
#print(np.shape(labels_train))
#print(np.shape(labels_val))

# TODO check shuffle and split on toy

clf = svm.SVC(kernel='linear')

# Training
clf.fit(betas_train, labels_train)

# Predict on val. data
pred = clf.predict(betas_val)
print(pred)

# Measure accuracy
print("Accuracy:", metrics.accuracy_score(labels_val, pred))
print("Precision:", metrics.precision_score(labels_val, pred)) # TP / (TP + FP)
print("Sensitivity:", metrics.recall_score(labels_val, pred)) # TP / (TP + FN)

tn, fp, fn, tp = confusion_matrix(labels_val, pred).ravel()

print("Specificity:", (tn / (tn + fp)))

