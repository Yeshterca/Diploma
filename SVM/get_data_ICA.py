# IMPORTS
import h5py
import numpy as np


def get_ica_data(directory):
    """
    Get the data obtained from ICA.
    :param directory: path to .mat file with temporal regression results
    :return: components_numbers = vector of sorted components' numbers,
             betas_all = 2D array of beta values (all patients, all components)
    """
    tempreg = h5py.File(directory, 'r')

    # Get component numbers
    components_table = tempreg['regressInfo/componentNumbers']
    components_array = np.squeeze(np.array(components_table))
    components_numbers = components_array.astype('int')

    # Get beta values
    betas_table = tempreg['regressInfo/regressionParameters']
    betas_array = np.array(betas_table)
    betas_all = betas_array.astype('float64')

    return components_numbers, betas_all


def select_components(no_patients, components_numbers, relevant_ics, betas_all, no_components):
    """
    Select only relevant components from the data.
    :param no_patients:
    :param components_numbers: vector of numbers of components after temporal sorting
    :param relevant_ics: vector of relevant IC numbers
    :param betas_all: 2D array of beta values (all ICs, all patients)
    :return: ic_numbers_selected = indices, ic_bets_selected = beta values of only relevant ICs (all patients)
    """

    ic_numbers_selected = []
    ic_betas_selected = np.zeros((no_components, no_patients))
    for i in range(0, no_components):
        idx = np.where(components_numbers == relevant_ics[i])[0][0]
        ic_numbers_selected.append(idx)
        ic_betas_selected[i, :] = betas_all[idx, :]

    return ic_numbers_selected, ic_betas_selected


def create_labels(no_hc, no_fes):
    """
    Create array of labels corresponding to data size.
    :param no_hc: number of healthy controls
    :param no_fes: number of schiz. patients
    :return: labels = array of 0/1 labels, 0 = HC, 1 = FES
    """
    hc_labels = np.zeros((1, no_hc), 'int')
    fes_labels = np.ones((1, no_fes), 'int')
    labels = np.squeeze(np.concatenate((hc_labels, fes_labels), axis=1))

    return labels


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


