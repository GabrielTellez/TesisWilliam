def GetCorrelations(array):

    corr = (array**2).mean(axis=1) - (array.mean(axis=1))**2

    return corr
