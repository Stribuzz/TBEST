"""

Welcome to the TBEST - TurBulence parameter EStimation from Telemetry Python Library

Author: Nuno Morujão

Feel free to use and expand this library to your own uses.

"""

import numpy as np
from scipy.optimize import leastsq
from fun_variance import nm,nZ_variance

def noiseVariance(ai):
    """

    Zernike coefficients noise variance computation by the
    temporal autocorrelation method.
    ai[modes,time]: 2d array with a sequence in time of Zc's
    from a set of modes.

    Author: Paulo Andrade
    Following the method described in Fusco 2004 (DOI: 10.1088/1464-4258/6/6/014)
    """

    nModes, nPs = ai.shape
    spfc = 1  # start point from center
    epfc = 4  # end point from center (center = nPs)
    poly_order = 6
    si2_noise = np.zeros((nModes))

    x = np.delete(np.arange(-epfc, epfc + 1), epfc)

    for iMode in range(nModes):

        c_rec = np.correlate(ai[iMode, :], ai[iMode, :], "full") / nPs
        c_points = np.concatenate((c_rec[nPs - epfc - 1:nPs - spfc], c_rec[nPs + spfc - 1:nPs + epfc]))
        c_fit = np.polyfit(x, c_points, poly_order)
        c_turb_0 = c_fit[poly_order]
        xx = c_rec[nPs - 1] - c_turb_0

        if (xx > 0):
            si2_noise[iMode] = xx

    return si2_noise


def ccCorrection(J, M, c, aiaj):

    """
    function si2_cc = ccCorrection1(J,M,c,aiaj)
    Computes the variance corrections for cross-coupling
    input:
    J    : number of modes in the reconstructor matrix
    M    : number of modes in the Zernike to slopes matrix
    c    : Cross-coupling matrix iH*Hr (J x K) x (K x M - J) -> (J x (M - J))
    aiaj : covariance matrix M x M
    output:
    si2_a   : variance correction for cca
    cca     : variance correction 1st term (see eq. 15 in report)
    cca_ct  : variance correction 2nd term (crossed term)

    K is number of lenslets accross the diameter

    Author: Paulo Andrade
    """
    cc = np.zeros(J);
    cc_ct = np.zeros(J);
    # si2_cc   = np.zeros(J);

    for ii in range(2, J + 1):  ## go through the modes (2,15) including 2 and 15.
        # print("ii = %4d "%(ii))
        for jj in range(J + 1, M - J + 1):  ## go through the non corrected modes (J,M).

            jc = jj - J  ## gives us a shifted version of the index (0, M - J).

            cc_ct[ii - 1] = cc_ct[ii - 1] + c[ii - 1, jc - 1] * aiaj[ii - 1, jj - 1]

            '''
            Summing over the cross correlation, we set the cross correlation of the piston to 0 by default and as such is not calculated in this code, 
            this can be included by changing the way we go through our matrix from 1 to 15 instead.
            '''

            for jl in range(J + 1, M - J + 1):  ## go through the non corrected modes (J,M) for the non crossed terms.
                jlc = jl - J
                cc[ii - 1] = cc[ii - 1] + c[ii - 1, jc - 1] * aiaj[jj - 1, jl - 1] * c[ii - 1, jlc - 1]

    return cc + 2 * cc_ct


def modesOfRadialOrder(n):
    """
    Returns the array with the Noll modes of radial order n
    Author: Paulo Andrade
    """
    return np.arange(n*(n + 1)/2 + 1 , (n + 1)*(n + 2)/2 + 1,dtype = int)


def std_vector(hRadOrd, lRadOrd, fitted_var):
    """
    :param hRadOrd: Highest radial order included in fit
    :param lRadOrd: Lowest radial order included in fit
    :param fitted_var: Fitted parameter - remaining and measurement noise removed
    :return: standard deviation of the radial orders included in the fit.
    Author: Nuno Morujão
    """

    stdv = np.zeros(hRadOrd - lRadOrd + 1)
    lIdx = 0 # last index
    fIdx = 0 # first index
    for ii in range(len(stdv)): # obtain the standard deviation of the radial orders
        fIdx = lIdx
        lIdx += len(modesOfRadialOrder(lRadOrd + ii))
        stdv[ii] = np.std(fitted_var[fIdx:lIdx])

    return stdv

def std_projection(hRadOrd, lRadOrd, std_vector):
    """
    :param hRadOrd: Highest radial order included in fit
    :param lRadOrd: Lowest radial order included in fit
    :param std_vector: standard deviations of radial orders
    :return: standard deviation vector for all azimuthal orders
    Author: Nuno Morujão
    """

    std = np.array([])
    for ii in range(hRadOrd - lRadOrd + 1):
        size = len(modesOfRadialOrder(lRadOrd + ii))
        std = np.append(std,np.ones(size)*std_vector[ii])

    return std

"""functions imported from OOMAO"""

def nModesFromRadialOrder(n):

    """
    nModeFromRadialOrder(n) returns the number of
    Zernike polynomials (n+1)(n+2)/2 up to a given radial order n
    """
    return int((n+1)*(n+2)/2);




class Turbulence_Estimator:

    """

    Turbulence_Estimator: Turbulence Parameters estimation - class to estimate
    r0 and L0. The Zernike coefficient (ZC) variances are fitted by the theoretical
    von Karman ZC variances.

    input:
    D:              telescope diameter
    modes:          vector with Noll modes to use in the fit
    modes_excluded: Particular Noll modes to be excluded from the fit
    ai2             vector with ZC variances
    si2_nn          vector with ZC noise variances
    si2_cc          vector with cross-coupling corrections to ZC variances
    lRadOrd         lowest radial order included in the fit
    hRadOrd         highest radial order included in the fit

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

    def __init__(self, D, modes, ai2, si2_nn, si2_cc=None, hRadOrd = 4,lRadOrd = 2, modes_excluded=np.array([])):

        self.D = D
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
        if (self.fitted_si2_cc is not None):
            self.fitted_si2_cc = si2_cc[[m - 1 for m in self.modes]]
            self.fitted_ai2 = self.fitted_ai2 - self.fitted_si2_cc

        # removes contribution of estimated measurement noise
        if (self.fitted_si2_nn is not None):
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
        self.stdv = std_vector(hRadOrd, lRadOrd, self.fitted_ai2)

        # Project the standard deviation to all Noll modes of the order
        self.std = std_projection(hRadOrd, lRadOrd, self.stdv)

        # fit function - theoretical zernike Variances as a function of turbulence parameters, r0 and L0
        af1 = lambda x, p: self.zernVariance(self.D, p, x)

        '''

        Here we define the fitting curve to be weighed by the standard deviation of the radial modes
        Serves as a more natural way of performing the least squares algorithm, avoiding the logarithm approach

        We need to recalculate the deviation every time we calculate the new fitted_ai2. As the points shift 
        in place.

        '''

        af2 = lambda p, x, y: (af1(x, p) - y) / self.std

        '''
        af2 uses 2 things;

        calculation of variances from current estimate of r0 and L0 and removes noise from it - Artificial

        subtracts the current fitted variances - Real

        Results in a residual of the model vs real data.
        '''

        self.fit = leastsq(af2, self.p0, args=(self.modes, self.fitted_ai2), full_output=1)

        self.tp = self.fit[0]

    def zernVariance(self, D, p, x):

        return nZ_variance(p[0], p[1], D, x[-1])[[m - 1 for m in x]]

