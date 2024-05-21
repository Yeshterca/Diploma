import numpy as np
import scipy.io as sio

rois_out = r'C:\Users\kajin\Documents\_\3\Thesis\ESO\eso\preprocess\num_rois_out_nudz.mat'

rois_out = sio.loadmat(rois_out)
rois_out = rois_out['num_roi_ko1']

print(rois_out)
np.save('rois_out_nudz', rois_out)


