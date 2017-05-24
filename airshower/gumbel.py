from __future__ import division

import numpy as np


def rand_gumbel(lgE, A, size=None, model='EPOS-LHC'):
    """
    Random Xmax values for given energy E [EeV] and mass number A
    See Manlio De Domenico et al., JCAP07(2013)050, doi:10.1088/1475-7516/2013/07/050
    Args:
        lgE (array): energy log10(E/eV)
        A (array): mass number
        model (string): hadronic interaction model
        size (int, optional): number of xmax values to create
    Returns:
        array of random Xmax values in [g/cm^2]
    """
    lE = lgE - 19
    lnA = np.log(A)
    D = np.array([np.ones_like(A), lnA, lnA**2])

    params = {
        'QGSJetII': {
            'mu': ((758.444, -10.692, -1.253), (48.892, 0.02, 0.179), (-2.346, 0.348, -0.086)),
            'sigma': ((39.033, 7.452, -2.176), (4.390, -1.688, 0.170)),
            'lambda': ((0.857, 0.686, -0.040), (0.179, 0.076, -0.0130))},
        'QGSJetII-04': {
            'mu': ((761.383, -11.719, -1.372), (57.344, -1.731, 0.309), (-0.355, 0.273, -0.137)),
            'sigma': ((35.221, 12.335, -2.889), (0.307, -1.147, 0.271)),
            'lambda': ((0.673, 0.694, -0.007), (0.060, -0.019, 0.017))},
        'Sibyll2.1': {
            'mu': ((770.104, -15.873, -0.960), (58.668, -0.124, -0.023), (-1.423, 0.977, -0.191)),
            'sigma': ((31.717, 1.335, -0.601), (-1.912, 0.007, 0.086)),
            'lambda': ((0.683, 0.278, 0.012), (0.008, 0.051, 0.003))},
        'EPOS1.99': {
            'mu': ((780.013, -11.488, -1.906), (61.911, -0.098, 0.038), (-0.405, 0.163, -0.095)),
            'sigma': ((28.853, 8.104, -1.924), (-0.083, -0.961, 0.215)),
            'lambda': ((0.538, 0.524, 0.047), (0.009, 0.023, 0.010))},
        'EPOS-LHC': {
            'mu': ((775.589, -7.047, -2.427), (57.589, -0.743, 0.214), (-0.820, -0.169, -0.027)),
            'sigma': ((29.403, 13.553, -3.154), (0.096, -0.961, 0.150)),
            'lambda': ((0.563, 0.711, 0.058), (0.039, 0.067, -0.004))}}
    param = params[model]

    p0, p1, p2 = np.dot(param['mu'], D)
    mu = p0 + p1 * lE + p2 * lE**2
    p0, p1 = np.dot(param['sigma'], D)
    sigma = p0 + p1 * lE
    p0, p1 = np.dot(param['lambda'], D)
    lambd = p0 + p1 * lE

    return mu - sigma * np.log(np.random.gamma(lambd, 1. / lambd, size=size))
