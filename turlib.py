"""
Welcome to the TurLib - Turbulence Library for the estimation of integral turbulence parameters from AO telemetry.

Author: Nuno Morujão & Paulo Andrade

Feel free to use and expand this library to your own uses.
"""

import numpy as np
from scipy.optimize import leastsq
from fun_variance import nm, nz_variance, nz_covariance


def noise_variance(ai):
    """

    Zernike coefficient noise variance computation by the
    temporal autocorrelation method.
    ai[modes,time]: 2d array with a sequence in time of Zc's
    from a set of modes.

    Author: Paulo Andrade
    Following the method described in Fusco 2004 (DOI: 10.1088/1464-4258/6/6/014)
    """

    n_modes, nps = ai.shape
    sp_fc = 1  # start point from center
    ep_fc = 4  # end point from center (center = nps)
    poly_order = 6
    si2_noise = np.zeros(n_modes)

    x = np.delete(np.arange(-ep_fc, ep_fc + 1), ep_fc)

    for iMode in range(n_modes):

        c_rec = np.correlate(ai[iMode, :], ai[iMode, :], "full") / nps
        c_points = np.concatenate((c_rec[nps - ep_fc - 1:nps - sp_fc], c_rec[nps + sp_fc - 1:nps + ep_fc]))
        c_fit = np.polyfit(x, c_points, poly_order)
        c_turb_0 = c_fit[poly_order]
        xx = c_rec[nps - 1] - c_turb_0

        if xx > 0:
            si2_noise[iMode] = xx

    return si2_noise


def cross_correction(n_rec_modes, m, c, ai_aj):
    """
    function si2_cc = cross_correction1(J,M,c,aiaj)
    Computes the variance corrections for cross-coupling
    input:
    n_rec_modes    : number of modes in the reconstructor matrix (J)
    m    : number of modes in the Zernike to slopes matrix
    c    : Cross-coupling matrix iH*Hr (J x K) x (K x M - J) -> (J x (M - J))
    ai_aj: covariance matrix M x M

    output:
    cc     : variance correction 1st term (see eq. 15 in report)
    cc_ct  : variance correction 2nd term (crossed term)

    Author: Paulo Andrade
    """
    cc = np.zeros(n_rec_modes)
    cc_ct = np.zeros(n_rec_modes)

    for ii in range(2, n_rec_modes + 1):  # go through the modes (2,15) including 2 and 15.
        for jj in range(n_rec_modes + 1, m - n_rec_modes + 1):  # go through the non-corrected modes (J,M).

            jc = jj - n_rec_modes  # gives us a shifted version of the index (0, M - J).

            cc_ct[ii - 1] = cc_ct[ii - 1] + c[ii - 1, jc - 1] * ai_aj[ii - 1, jj - 1]

            '''
            Summing over the cross correlation, we set the cross correlation of the piston to 0 by default and as such 
            is not calculated in this code, this can be included by changing the way we go through our matrix 
            from 1 to 15 instead.
            '''

            for jl in range(n_rec_modes + 1, m - n_rec_modes + 1):
                # go through the non-corrected modes (J,M) for the non-crossed terms.
                jlc = jl - n_rec_modes
                cc[ii - 1] = cc[ii - 1] + c[ii - 1, jc - 1] * ai_aj[jj - 1, jl - 1] * c[ii - 1, jlc - 1]

    return cc + 2 * cc_ct


def modes_of_radial_order(n):
    """
    Returns the array with the Noll modes of radial order n
    Author: Paulo Andrade
    """
    return np.arange(n * (n + 1) / 2 + 1, (n + 1) * (n + 2) / 2 + 1, dtype=int)


def std_vector(h_rad_ord, l_rad_ord, fitted_var):
    """
    :param h_rad_ord: Highest radial order included in fit
    :param l_rad_ord: Lowest radial order included in fit
    :param fitted_var: Fitted parameter - remaining and measurement noise removed
    :return: standard deviation of the radial orders included in the fit.
    Author: Nuno Morujão
    """

    std_v = np.zeros(h_rad_ord - l_rad_ord + 1)
    l_idx = 0  # last index
    for ii in range(len(std_v)):  # obtain the standard deviation of the radial orders
        f_idx = l_idx
        l_idx += len(modes_of_radial_order(l_rad_ord + ii))
        std_v[ii] = np.std(fitted_var[f_idx:l_idx])

    return std_v


def std_projection(h_rad_ord, l_rad_ord, standard_dev_vector):
    """
    :param h_rad_ord: Highest radial order included in fit
    :param l_rad_ord: Lowest radial order included in fit
    :param standard_dev_vector: standard deviations of radial orders
    :return: standard deviation vector for all azimuthal orders
    Author: Nuno Morujão
    """

    std = np.array([])
    for ii in range(h_rad_ord - l_rad_ord + 1):
        size = len(modes_of_radial_order(l_rad_ord + ii))
        std = np.append(std, np.ones(size) * standard_dev_vector[ii])

    return std


"""functions imported from OOMAO"""


def n_modes_from_radial_order(n):
    """
    nModeFromRadialOrder(n) returns the number of
    Zernike polynomials (n+1)(n+2)/2 up to a given radial order n
    """
    return int((n + 1) * (n + 2) / 2)


def zernike_variance(d, p, x):
    return nz_variance(p[0], p[1], d, x[-1])[[m - 1 for m in x]]


"""functions for iterative estimation of the turbulence parameters"""


def iterative_estimator(d, modes, ai2, noise_estimate, n_rec_modes, m, c_mat, n_iter=5, full_vector=False):
    """
    :param d: diameter of telescope
    :param modes: modes [Noll convention] included in the fit
    :param ai2: pseudo-open loop variances
    :param noise_estimate: estimation of noise from noise_variance function (check documentation)
    :param n_rec_modes: Number of modes included in the reconstruction
    :param m: Total size of the c_mat matrix
    :param c_mat: Cross-talk matrix - specific to your system.
    :param n_iter: Number of iterations of the algorithm
    :param full_vector: Full vector permits saving all turbulence parameter estimates
    :return: Returns the turbulence parameters (r0,l0) fitted from the turbulence estimator class
    """

    # initial estimation without cross talk correction
    tp = TurbulenceEstimator(d, modes, ai2, si2_nn=noise_estimate)

    r0 = tp.tp[0]
    l0 = tp.tp[1]

    if full_vector:
        r0_vector = np.zeros(n_iter + 1)
        l0_vector = np.zeros(n_iter + 1)

        r0_vector[0] = r0
        l0_vector[0] = l0

    for vv in range(n_iter):
        # Calculation of the remaining error contributions
        aiaj_0 = nz_covariance(r0, l0, d, m)
        si2_cc_1 = cross_correction(n_rec_modes, m + n_rec_modes, c_mat, aiaj_0)

        tp = TurbulenceEstimator(d, modes, ai2, si2_nn=noise_estimate, si2_cc=si2_cc_1)

        r0 = tp.tp[0]
        l0 = tp.tp[1]

        if full_vector:
            r0_vector[vv + 1] = r0
            l0_vector[vv + 1] = l0

    if full_vector:
        return r0_vector, l0_vector, tp.fitted_ai2

    if not full_vector:
        return r0, l0, tp.fitted_ai2


def uncertainty_estimator(d, modes, ai2, noise_estimate, n_rec_modes, m, c_mat, l_rad_ord=2,
                          h_rad_ord=4, n_samples=50, n_iter=5):
    """
    :param n_iter: Number of iterations for the correction of the variances - By default it is 5
    :param noise_estimate: Estimation of noise from noise_variance model
    :param d: diameter of telescope
    :param modes: modes to be used in the fit
    :param ai2: reconstructed pseudo-open modes
    :param n_rec_modes: maximum mode reconstructed
    :param m: total number of modes included in the matrix
    :param c_mat: cross-talk matrix
    :param l_rad_ord: lowest radial order fit
    :param h_rad_ord: highest radial order fit
    :param n_samples: number of samples used to calculate the uncertainty (50 by default)
    :return: Estimated turbulence parameters with an added uncertainty estimate (r0, u(r0)) (L0, u(L0))
    """

    r0_vector = np.zeros(n_samples)
    l0_vector = np.zeros(n_samples)

    corr_modes = iterative_estimator(d, modes, ai2, noise_estimate, n_rec_modes, m, c_mat, n_iter=n_iter)[-1]
    standard_deviations = std_projection(h_rad_ord, l_rad_ord, std_vector(h_rad_ord, l_rad_ord, corr_modes))
    modal_vector = np.zeros(n_rec_modes)

    for kk in range(n_samples):
        modal_vector[n_modes_from_radial_order(l_rad_ord - 1):] = np.random.normal(corr_modes, standard_deviations)
        # We want to study the variability of the residual of the compensation
        r0, l0 = TurbulenceEstimator(d, modes, modal_vector, si2_nn=None, si2_cc=None).tp

        r0_vector[kk] = r0
        l0_vector[kk] = l0

    mean_r0 = np.nanmean(r0_vector)
    mean_l0 = np.nanmean(l0_vector)
    # Extra sqrt factor so to conform to an error around a mean
    error_r0 = np.nanstd(r0_vector)/np.sqrt(n_samples)
    error_l0 = np.nanstd(l0_vector)/np.sqrt(n_samples)

    return [mean_r0, error_r0], [mean_l0, error_l0]


def full_uncertainty_estimator(d, modes, ai2, noise_estimate, n_rec_modes, m, c_mat, l_rad_ord=2, h_rad_ord=4,
                               n_samples=50, n_iter=5):
    """
    :param n_iter: Number of iterations of the iterative algorithm
    :param noise_estimate: Estimation of noise from noise_variance model
    :param d: diameter of telescope
    :param modes: modes to be used in the fit
    :param ai2: reconstructed pseudo-open modes
    :param n_rec_modes: maximum mode reconstructed
    :param m: total number of modes included in the matrix
    :param c_mat: cross-talk matrix
    :param l_rad_ord: lowest radial order fit
    :param h_rad_ord: highest radial order fit
    :param n_samples: number of samples used to calculate the uncertainty (50 by default)
    :return: Estimated turbulence parameters with an added uncertainty estimate (r0, u(r0)) (L0, u(L0))
    """

    r0_vector = np.zeros(n_samples)
    l0_vector = np.zeros(n_samples)
    s_idx = n_modes_from_radial_order(l_rad_ord - 1)
    standard_deviations = std_projection(h_rad_ord, l_rad_ord, std_vector(h_rad_ord, l_rad_ord, ai2[3:]))
    r0_i, l0_i = iterative_estimator(d, modes, ai2, noise_estimate, n_rec_modes, m, c_mat, n_iter=n_iter)[:2]
    modal_vector = np.zeros(n_rec_modes)

    for kk in range(n_samples):
        modal_vector[s_idx:] = np.random.normal(ai2[s_idx:], standard_deviations)
        r0, l0 = iterative_estimator(d, modes, modal_vector, noise_estimate, n_rec_modes, m, c_mat, n_iter=n_iter)[:2]

        r0_vector[kk] = r0
        l0_vector[kk] = l0

    # Extra sqrt factor so to conform to an error around a mean
    error_r0 = np.nanstd(r0_vector) / np.sqrt(n_samples)
    error_l0 = np.nanstd(l0_vector) / np.sqrt(n_samples)

    return [r0_i, error_r0], [l0_i, error_l0], r0_vector


class TurbulenceEstimator:
    """

    Turbulence_Estimator: Turbulence Parameters estimation - class to estimate
    r0 and L0. The Zernike coefficient (ZC) variances are fitted by the theoretical
    von Karman ZC variances.

    input:
    d:              telescope diameter
    modes:          vector with Noll modes to use in the fit
    modes_excluded: Particular Noll modes to be excluded from the fit
    ai2             vector with ZC variances
    si2_nn          vector with ZC noise variances
    si2_cc          vector with cross-coupling corrections to ZC variances
    l_rad_ord       the lowest radial order included in the fit
    h_rad_ord       the highest radial order included in the fit

    output:
    fit             scipy.optimize.leastsq output
    tp              vector with parameter estimates
    sp              vector with parameter estimates uncertainty
    fitted_ai2      fitted variances
    fitted_si2_nn   noise variances of fitted modes
    fitted_si2_cc   cross-coupling corrections for the fitted modes
    fitted_nn       fitted modes radial order
    fitted_mm       fitted modes azimuthal order

    Authors: Nuno Morujão, Paulo Andrade

    """

    def __init__(self, d, modes, ai2, si2_nn, si2_cc=None, h_rad_ord=4, l_rad_ord=2, modes_excluded=np.array([])):

        self.D = d
        self.modes = modes  # (numbering assumes index 1 as piston)
        self.modes_excluded = modes_excluded
        for ime in range(modes_excluded.size):
            self.modes = self.modes[self.modes != self.modes_excluded[ime]]

        self.fitted_ai2 = ai2[[m - 1 for m in self.modes]]
        self.fitted_si2_cc = si2_cc
        self.fitted_si2_nn = si2_nn
        self.fitted_nn = [nm(m)[0] for m in self.modes]
        self.fitted_mm = [nm(m)[1] for m in self.modes]

        # removes contribution of estimated remaining noise
        if self.fitted_si2_cc is not None:
            self.fitted_si2_cc = si2_cc[[m - 1 for m in self.modes]]
            self.fitted_ai2 = self.fitted_ai2 - self.fitted_si2_cc

        # removes contribution of estimated measurement noise
        if self.fitted_si2_nn is not None:
            self.fitted_si2_nn = si2_nn[[m - 1 for m in self.modes]]
            self.fitted_ai2 = self.fitted_ai2 - self.fitted_si2_nn

        '''

        From here we remove the first version of the of the noise estimate.

        '''

        # Parameters initial guess - we randomize the positions in order not to induce bias in the fitting
        # Recommended small initial parameters - from the chi squared map of the algorithm for an on-sky sample

        self.p0 = np.array([.01 + np.random.rand() * .01, np.random.random() * 4 + 1])

        # weight vector for the chi square

        # Obtain the standard deviations within radial orders
        self.std_v = std_vector(h_rad_ord, l_rad_ord, self.fitted_ai2)

        # Project the standard deviation to all Noll modes of the order
        self.std = std_projection(h_rad_ord, l_rad_ord, self.std_v)

        # fit function - theoretical zernike Variances as a function of turbulence parameters, r0 and L0
        def af1(x, p):
            return zernike_variance(self.D, p, x)

        '''

        Here we define the fitting curve to be weighed by the standard deviation of the radial modes
        Serves as a more natural way of performing the least squares algorithm, avoiding the logarithm approach

        We need to recalculate the deviation every time we calculate the new fitted_ai2. As the points shift 
        in place.

        '''

        def af2(p, x, y):
            return (af1(x, p) - y) / self.std

        '''
        af2 uses 2 things;

        calculation of variances from current estimate of r0 and L0 and removes noise from it - Artificial

        subtracts the current fitted variances - Real

        Results in a residual of the model vs real data.
        '''

        self.fit = leastsq(af2, self.p0, args=(self.modes, self.fitted_ai2), full_output=True)

        self.tp = self.fit[0]
