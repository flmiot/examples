import numpy as np
import scipy.optimize as optim
import scipy.special

""" Exponentially modified Gaussian (EMG) functions --
    the convolution between an exponential decay and a Gaussian function --
    allow us to model an time-resolved decay process -- e.g. an pulse-excitation
    -- measured via an instrument with finite instrument response function (IRF)
"""

class Model_EMF_GF(object):
    """ Define an exponentially modified Gaussian function (EMG) base class """

    def c(self, t, k, mu, sigma):
        return 0.5 * np.exp(-k*(t-mu-(sigma**2*k)/2)) * (1+scipy.special.erf((t-mu-sigma**2*k)/(np.sqrt(2)*sigma)))

    def es1_0(self, t, sigma):
        return 1

    def s(self, t, sigma):
        return self.es1_0(t, sigma) * 0.5*scipy.special.erfc(-t/np.sqrt(2)/sigma)

class Model_1DEC_GF(Model_EMF_GF):
    """ Implement a 2-exponential decay model for lmfit fitting """

    def __init__(self, **param_guesses):
        self.guesses = param_guesses

    def es1(self, t, k1, sigma, a1):
        mu = 0
        return a1 * self.es1_0(t, sigma)*self.c(t, k1, mu, sigma)

    def es2(self, t, k1, sigma, a2):
        mu = 0
        return a2*(self.s(t, sigma) - self.c(t, k1, mu, sigma))

    def model(self, t, k1, a1, a2, sigma, x0):
        return self.es1(t-x0, k1, sigma, a1) + self.es2(t-x0, k1, sigma, a2)

    def make_lmfit_parameters(self, data):
        params = lmfit.Parameters()
        l = data.shape[0]

        for i in range(l):
            params.add(f'a1_{i+1}', value=1, min=-100, max=100)
            params.add(f'a2_{i+1}', value=1, min=-100, max=100)

        params.add('t1', value=1.7, min=0.15, max=10)
        params.add('k1', expr =  '1/t1')
        params.add(f'irf_fwhm', value=0.112, min=0.05, max=0.5, vary = False)
        params.add(f'irf_sigma', expr = 'irf_fwhm/(2*sqrt(2*log(2)))')
        params.add(f'x0', value=-0.1, vary = False)


        #params.add('area1', value=0, vary = False, expr = '+'.join([f'a1_{i+1}' for i in range(l)]))
        #params.add('area2', value=0, vary = False, expr = '+'.join([f'a2_{i+1}' for i in range(l)]))

        for pname, pval in self.guesses.items():
            params[pname].value = pval

        return params

    def model_dataset(self, params, i, x):
        """Calculate a sum of decays from parameters for data set."""
        a1 = params[f'a1_{i+1}']
        a2 = params[f'a2_{i+1}']
        sigma = params['irf_sigma']
        k1 = params['k1']
        x0 = params['x0']
        return self.model(x, k1, a1, a2, sigma, x0)

    def objective(self, params, x, data, data_err = None):
        """Calculate total residual for fits to several data sets."""
        nenergies, _ = data.shape
        resid = 0.0*data[:]
        # make residual per data set
        if data_err is None:
            for i in range(nenergies):
                resid[i] = data[i] - self.model_dataset(params, i, x)

        else:
            for i in range(nenergies):
                resid[i] = (data[i] - self.model_dataset(params, i, x))/data_err[i]

        return resid.flatten()


class Model_2DEC_GF(Model_EMF_GF):
    """ Implement a 2-exponential decay model for lmfit fitting """

    def __init__(self, **param_guesses):
        self.guesses = param_guesses

    def es1(self, t, k1, sigma, a1):
        mu = 0
        return a1 * self.es1_0(t, sigma)*self.c(t, k1, mu, sigma)

    def es2(self, t, k1, k2, sigma, a2):
        mu = 0
        return a2*self.es1_0(t, sigma)*k1 / (k2-k1) * (self.c(t, k1, mu, sigma)-self.c(t, k2, mu, sigma))

    def es3(self, t, k1, k2, sigma, a3):
        mu = 0
        return a3*(self.s(t, sigma) - (k2*self.c(t, k1, mu, sigma)- k1*self.c(t, k2, mu, sigma))/(k2-k1))

    def model(self, t, k1, k2, a1, a2, a3, sigma, x0):
        return self.es1(t-x0, k1, sigma, a1) + self.es2(t-x0, k1, k2, sigma, a2) + self.es3(t-x0, k1, k2, sigma, a3)

    def make_lmfit_parameters(self, data):
        params = lmfit.Parameters()
        l = data.shape[0]
        for i in range(l):
            params.add(f'a1_{i+1}', value=-20, min=-100, max=100)
            params.add(f'a2_{i+1}', value=-20, min=-100, max=100)
            params.add(f'a3_{i+1}', value=36, min=-100, max=100)

        params.add(f't1', value=1.4, min=0.1, max=3, vary = True)
        params.add(f't2', value=8.3, min=0.1, max=40, vary = True)
        params.add('k1', expr =  '1/t1')
        params.add('k2', expr =  '1/t2')
        params.add(f'irf_fwhm', value=0.112, min=0.05, max=0.5, vary = False)
        params.add(f'irf_sigma', expr = 'irf_fwhm/(2*sqrt(2*log(2)))')
        params.add(f'x0', value=-0.1, vary = False)


        params.add('area1', value=0, vary = False, expr = '+'.join([f'a1_{i+1}' for i in range(l)]))
        params.add('area2', value=0, vary = False, expr = '+'.join([f'a2_{i+1}' for i in range(l)]))
        params.add('area3', value=0, vary = False, expr = '+'.join([f'a3_{i+1}' for i in range(l)]))

        for pname, pval in self.guesses.items():
            params[pname].value = pval

        return params

    def model_dataset(self, params, i, x):
        """Calculate a sum of decays from parameters for data set."""
        a1 = params[f'a1_{i+1}']
        a2 = params[f'a2_{i+1}']
        a3 = params[f'a3_{i+1}']
        sigma = params['irf_sigma']
        k1 = params['k1']
        k2 = params['k2']
        x0 = params['x0']
        return self.model(x, k1, k2, a1, a2, a3, sigma, x0)

    def objective(self, params, x, data, data_err = None):
        """Calculate total residual for fits to several data sets."""
        nenergies, _ = data.shape
        resid = 0.0*data[:]

        # make residual per data set
        if data_err is None:
            for i in range(nenergies):
                resid[i] = data[i] - self.model_dataset(params, i, x)

        else:
            for i in range(nenergies):
                resid[i] = (data[i] - self.model_dataset(params, i, x))/data_err[i]

        return resid.flatten()
