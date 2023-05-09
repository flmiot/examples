import lmfit
from models import Model_1DEC_GF

# Provide data here
fit_delay = None
fit_intensity = None
fit_intensity_err = None

if __name__ == '__main__':
    model = Model_1DEC_GF()
    params = model.make_lmfit_parameters(fit_intensity)

    out1 = lmfit.minimize(
        model.objective,
        params,
        args=(fit_delay, fit_intensity, fit_intensity_err), scale_covar = False)
