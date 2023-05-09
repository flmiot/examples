import numpy as np

def multi_pv(number_of_f):
    """ Multi-component pseudo-voigt function. *param_dict* should be a dict of parameter values """

    def f(x, a, b, c, d):
        """ a : Total integrated area
            b : mixing ration gaussian - lorentzian
            c : width
            d : position
        """
        return a*((1-b)/(np.pi*c*(1+((x-d)/c)**2))+b*(np.sqrt(2*np.log(2)))/(c*np.sqrt(2*np.pi))*np.exp(-1*((x-d)**2/(2*(c/(np.sqrt(2*np.log(2))))**2))))

    def make_multiple_f(f, number):
        def f_(x, *arguments):
            this_length = number
            if len(arguments) != this_length * 4:
                fmt = 'Wrong number of arguments: Need x and {} parameters'.format(this_length * 4)
                raise ValueError(fmt)

            for i in range(this_length):
                sl = slice(int(i*4), int((i+1)*4))
                if i == 0:
                    y = f(x, *arguments[sl])
                else:
                    y += f(x, *arguments[sl])

            return y

        return f_

    return make_multiple_f(f, number_of_f)
