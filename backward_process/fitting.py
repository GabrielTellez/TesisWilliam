from scipy.optimize import curve_fit

def ExponentialFitting(x,y):

    #finds the critical exponent

    f = lambda x, a: x**a

    a, sigma_a = curve_fit(f, x, y)

    return a, sigma_a

#Testing

#x = np.arange(10)
#y = np.sqrt(x)

#exponential_fitting(x,y)