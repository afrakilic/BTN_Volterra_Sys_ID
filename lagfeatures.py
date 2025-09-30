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

    # Convert to numpy arrays
    input_ = np.asarray(input_)
    tinput = np.asarray(tinput)
    output = np.asarray(output)
    toutput = np.asarray(toutput)
    inlags = np.asarray(inlags)
    outlags = np.asarray(outlags)

    # Determine maximum lag
    max_lag = max(inlags.max(), outlags.max())

    # Calculate how many samples we can use
    ending = len(input_) - max_lag
    endingt = len(tinput) - max_lag

    if ending <= 0 or endingt <= 0:
        raise ValueError("Not enough data points for the specified lags.")

    # Preallocate arrays
    N = ending  # number of final usable samples
    u = np.zeros((N, len(inlags)))
    uv = np.zeros((N, len(inlags)))
    y = np.zeros((N, len(outlags)))
    yv = np.zeros((N, len(outlags)))

    # ---- Generate features using input lags ----
    for l, lag in enumerate(inlags):
        # Slice to match MATLAB inclusive indexing
        # MATLAB: input(end - lag - ending + 1 : end - lag)
        u[:, l] = input_[-lag - ending : -lag]
        uv[:, l] = tinput[-lag - endingt : -lag]

    # ---- Generate features using output lags ----
    for l, lag in enumerate(outlags):
        if lag == 0:
            # slice ending elements, length = ending
            y[:, l] = output[-ending:]
            yv[:, l] = toutput[-endingt:]
        else:
            y[:, l] = output[-(lag + ending):-lag]
            yv[:, l] = toutput[-(lag + endingt):-lag]

    # ---- Combine features ----
    # Reverse output lags and exclude the current output (first column in MATLAB)
    featurez = np.hstack([y[:, ::-1][:, 0:], u])
    zeta = y[:, 0]  # target variable is the most recent output

    tfeaturez = np.hstack([yv[:, ::-1][:, 0:], uv])
    yt = yv[:, 0]  # target time vector

    return featurez, zeta, tfeaturez, yt
