import numpy as np

def lagfeatures(input_, tinput, output, toutput, inlags, outlags):
    """
    Generate lagged features for input and output signals.
    
    Parameters:
    input_ : array-like, input time series
    tinput : array-like, time vector for input
    output : array-like, output time series
    toutput: array-like, time vector for output
    inlags  : list/array of input lags
    outlags : list/array of output lags
    
    Returns:
    featurez : array, lagged features of inputs and outputs
    zeta     : array, target variable
    tfeaturez: array, lagged features of input and output time vectors
    yt       : array, target time vector
    """

    input_ = np.asarray(input_)
    tinput = np.asarray(tinput)
    output = np.asarray(output)
    toutput = np.asarray(toutput)

    # Calculate ending indices
    ending = len(input_) - max(outlags[-1], inlags[-1]) - 1
    endingt = len(tinput) - max(outlags[-1], inlags[-1]) - 1

    # Generate features using input lags
    u = np.column_stack([input_[len(input_)-inlag-ending : len(input_)-inlag] for inlag in inlags])
    uv = np.column_stack([tinput[len(tinput)-inlag-endingt : len(tinput)-inlag] for inlag in inlags])

    # Generate features using output lags
    y = np.column_stack([output[len(output)-outlag-ending : len(output)-outlag] for outlag in outlags])
    yv = np.column_stack([toutput[len(toutput)-outlag-endingt : len(toutput)-outlag] for outlag in outlags])

    # Features: lagged outputs (except first column) + lagged inputs
    featurez = np.hstack([y[:, ::-1][:, 1:], u])
    zeta = y[:, 0]

    tfeaturez = np.hstack([yv[:, ::-1][:, 1:], uv])
    yt = yv[:, 0]

    return featurez, zeta, tfeaturez, yt
