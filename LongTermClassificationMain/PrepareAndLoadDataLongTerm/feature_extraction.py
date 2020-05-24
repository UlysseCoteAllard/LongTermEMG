import math
import pywt
import sampen
import numpy as np
from scipy import stats, spatial


# Util function cmp (needed as we are in Python 3)
def cmp(a, b):
    return bool(a > b) - bool(a < b)


def extract_features(vector):
    features = []
    for c in range(len(vector)):
        # plt.plot(np.asarray(vector[c]))
        # plt.show()
        #    features_data.append(getApEn(vector[c]))
        features.extend(getAR(vector[c], 4))
        features.extend(getCC(vector[c], 4))
        features.append(getDASDV(vector[c]))
        #    features_data.append(getFuzzyEn(vector))
        features.extend(getHIST(vector, 3))

        #    features_data.append(getHOMYOP(vector))
        #    features_data.append(getHOSSC(vector))
        #    features_data.append(getHOWAMP(vector))
        #    features_data.append(getHOZC(vector))

        features.append(getIEMG(vector[c]))
        features.extend(getIQR(vector[c]))
        features.append(getLD(vector[c]))
        #    features_data.append(getLS(vector)) # LOOKUP LMOM function to implement this <--
        features.extend(getMAVFD(vector[c]))
        features.append(getMAVFDn(vector[c]))
        features.append(getMAV(vector[c]))
        features.append(getMAVSD(vector[c]))
        features.append(getMAVSDn(vector[c]))
        features.extend(getMAVSLP(vector[c], 2))
        features.append(getMDF(vector[c], 1000))
        features.append(getMMAV1(vector[c]))
        features.append(getMMAV2(vector[c]))
        features.append(getMNF(vector[c], 1000))
        features.append(getMNP(vector[c]))
        features.append(getMPK(vector[c]))
        features.append(getMSR(vector[c]))
        features.append(getMYOP(vector[c], 1))
        features.append(getRANGE(vector[c]))
        features.append(getRMS(vector[c]))
        # features_data.append(getSampEn(vector[c]))
        features.append(getSKEW(vector[c]))
        features.append(getSM(vector[c], 2, 1000))
        features.append(getSSC(vector[c], 0.01))
        features.append(getSSI(vector[c]))
        features.append(getSTD(vector[c]))
        features.append(getTM(vector[c]))
        features.append(getTTP(vector[c]))
        features.append(getVAR(vector[c]))
        features.append(getWAMP(vector[c], 0.1))
        features.append(getWL(vector[c]))
        features.append(getZC(vector[c], 0.1))
    return features


# Funtion adapted from the lmoments package by Sam Gillespie (https://pydoc.net/lmoments/0.2.2/)
def _comb(n, k):
    if (k > n) or (n < 0) or (k < 0):
        return 0
    val = 1
    for j in range(min(k, n - k)):
        val = (val * (n - j)) // (j + 1)
    return val


# Funtion adapted from the lmoments package by Sam Gillespie (https://pydoc.net/lmoments/0.2.2/)
def getL_scale(x):
    x = sorted(x)
    n = len(x)

    comb1 = range(0, n)
    comb2 = range(n - 1, -1, -1)

    coefl2 = 0.5 * 1.0 / _comb(n, 2)
    xtrans = []
    for i in range(0, n):
        coeftemp = comb1[i] - comb2[i]
        xtrans.append(coeftemp * x[i])

    l2 = coefl2 * sum(xtrans)
    return l2


# New function
def getDAR(vector, order=4):
    # Get the first difference of the vector
    vector_diff = np.diff(vector)
    # Calculate the AR coefficient on it
    return getAR(vector_diff, order=4)


# New function
def getAR(vector, order=4):
    # Using Levinson Durbin prediction algorithm, get autoregressive coefficients
    # Square signal
    vector = np.asarray(vector)
    R = [vector.dot(vector)]
    if R[0] == 0:
        return [1] + [0] * (order - 2) + [-1]
    else:
        for i in range(1, order + 1):
            r = vector[i:].dot(vector[:-i])
            R.append(r)
        R = np.array(R)
        # step 2:
        AR = np.array([1, -R[1] / R[0]])
        E = R[0] + R[1] * AR[1]
        for k in range(1, order):
            if (E == 0):
                E = 10e-17
            alpha = - AR[:k + 1].dot(R[k + 1:0:-1]) / E
            AR = np.hstack([AR, 0])
            AR = AR + alpha * AR[::-1]
            E *= (1 - alpha ** 2)
        return AR


# New function
def getCC(vector, order=4):
    AR = getAR(vector, order)
    cc = np.zeros(order + 1)
    cc[0] = -1 * AR[0]  # issue with this line
    if order > 2:
        for p in range(2, order + 2):
            for l in range(1, p):
                cc[p - 1] = cc[p - 1] + (AR[p - 1] * cc[p - 2] * (1 - (l / p)))

    return cc


# New function
def getDASDV(vector):
    vector = np.asarray(vector)
    return np.lib.scimath.sqrt(np.mean(np.diff(vector)))


# New function
def getHIST(vector, threshold_nmbr_of_sigma, bins=3):
    # calculate sigma of signal
    sigma = np.std(vector)
    mean = np.std(vector)
    threshold = threshold_nmbr_of_sigma * sigma
    hist, bin_edges = np.histogram(vector, bins=bins, range=(mean - threshold, mean + threshold))
    return hist


# New function
def getIEMG(vector):
    vector = np.asarray(vector)
    return np.sum(np.abs(vector))


# New function
# Fractal dimension using Box Counting
def getBC(vector):
    k_max = int(np.floor(np.log2(len(vector)))) - 1

    Nr = np.zeros(k_max)
    r = np.zeros(k_max)
    for k in range(0, k_max):
        r[k] = 2 ** (k + 1)
        curve_box = int(np.floor(len(vector) / r[k]))
        box_r = np.zeros(curve_box)
        for i in range(curve_box):
            max_dat = np.max(vector[int(r[k] * i):int(r[k] * (i + 1))])
            min_dat = np.min(vector[int(r[k] * i):int(r[k] * (i + 1))])

            box_r[i] = np.ceil((max_dat - min_dat) / r[k])
        Nr[k] = np.sum(box_r)

    bc_poly = np.polyfit(np.log2(1 / r), np.log2(Nr), 1)
    return bc_poly[0]


# New function (Maximum Fractal Length)
def getMFL(vector):
    return np.log10(np.sum(abs(np.diff(vector))))


# New function
def getIQR(vector):
    vector = np.asarray(vector)
    vector.sort()
    return [vector[int(round(vector.shape[0] / 4))], vector[int(round(vector.shape[0] * 3 / 4))]]


# New function
def getLD(vector):
    vector = np.asarray(vector)
    return np.exp(np.mean(np.log(np.abs(vector) + 1)))


# New function
def getMAVFD(vector):
    vector = np.diff(np.asarray(vector))
    total_sum = 0
    for i in range(len(vector)):
        total_sum += abs(vector[i])
    return (total_sum / vector.shape).tolist()


# New function
def getMAVFDn(vector):
    vector = np.asarray(vector)
    std = np.std(vector)
    return np.mean(np.abs(np.diff(vector))) / std


# New function
def getMAVSD(vector):
    vector = np.asarray(vector)
    return np.mean(np.abs(np.diff(np.diff(vector))))


# New function
def getMAVSDn(vector):
    vector = np.asarray(vector)
    std = np.std(vector)
    return np.mean(np.abs(np.diff(np.diff(vector)))) / std


def getMAVSLP(vector, segment=2):
    vector = np.asarray(vector)
    m = int(round(vector.shape[0] / segment))
    mav = []
    mavslp = []
    for i in range(0, segment):
        mav.append(np.mean(np.abs(vector[i * m:(i + 1) * m])))
    for i in range(0, segment - 1):
        mavslp.append(mav[i + 1] - mav[i])
    return mavslp


# Ulysse's function
def getMAV(vector):
    return np.mean(np.abs(vector))


# New function
def getMDF(vector, fs=1000):
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0] / 2))]
    # f = np.fft.fftfreq(vector.shape[-1])
    POW = spec * np.conj(spec)
    totalPOW = np.sum(POW)
    medfreq = 0
    for i in range(0, vector.shape[0]):
        if np.sum(POW[0:i]) > 0.5 * totalPOW:
            medfreq = fs * i / vector.shape[0]
            break
    return medfreq


# Ulysse function
def getMMAV1(vector):
    vector_array = np.array(vector)
    total_sum = 0.0
    for i in range(0, len(vector_array)):
        if ((i + 1) < 0.25 * len(vector_array) or (i + 1) > 0.75 * len(vector_array)):
            w = 0.5
        else:
            w = 1.0
        total_sum += abs(vector_array[i] * w)
    return total_sum / len(vector_array)


def getMMAV2(vector):
    total_sum = 0.0
    vector_array = np.array(vector)
    for i in range(0, len(vector_array)):
        if ((i + 1) < 0.25 * len(vector_array)):
            w = ((4.0 * (i + 1)) / len(vector_array))
        elif ((i + 1) > 0.75 * len(vector_array)):
            w = (4.0 * ((i + 1) - len(vector_array))) / len(vector_array)
        else:
            w = 1.0
        total_sum += abs(vector_array[i] * w)
    return total_sum / len(vector_array)


# New function
def getMNF(vector, fs=1000):
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    f = np.fft.fftfreq(vector.shape[-1]) * fs
    spec = spec[0:int(round(spec.shape[0] / 2))]
    f = f[0:int(round(f.shape[0] / 2))]
    POW = spec * np.conj(spec)

    return np.sum(POW * f) / sum(POW)


# New function
def getMNP(vector):
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0] / 2))]
    POW = spec * np.conj(spec)
    return np.sum(POW) / POW.shape[0]


# New function
def getMPK(vector):
    vector = np.asarray(vector)
    return vector.max()


# New function
def getMSR(vector):
    vector = np.asarray(vector)
    return (np.abs(np.mean(np.lib.scimath.sqrt(vector))))


# New function
def getMYOP(vector, threshold=1.):
    vector = np.asarray(vector)
    return np.sum(np.abs(vector) >= threshold) / float(vector.shape[0])


# New function
def getRANGE(vector, filsize=2):
    vector = np.asarray(vector)
    return vector.max() - vector.min()


# New function
def getRMS(vector):
    vector = np.asarray(vector)
    return np.sqrt(np.mean(np.square(vector)))


# Ulysse function
def getSampEn(vector, m=2, r_multiply_by_sigma=.2):
    vector_np = np.asarray(vector)
    r = r_multiply_by_sigma * np.std(vector_np)
    results = sampen.sampen2(data=vector, mm=m, r=r)
    results_SampEN = []
    for x in np.array(results)[:, 1]:
        if x is not None:
            results_SampEN.append(x)
        else:
            results_SampEN.append(-100.)
    return list(results_SampEN)


# New function
def getSKEW(vector):
    vector = np.asarray(vector)
    return stats.skew(vector)


# New function
def getKURT(vector):
    vector = np.asarray(vector)
    return stats.kurtosis(vector)


# New function
def getSM(vector, order=2, fs=1000):
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0] / 2))]
    f = np.fft.fftfreq(vector.shape[-1]) * fs
    f = f[0:int(round(f.shape[0] / 2))]
    POW = spec * np.conj(spec)
    return np.sum(POW.dot(np.power(f, order)))


# Ulysse function
def getSSC(vector, threshold=0.1):
    vector = np.asarray(vector)
    slope_change = 0
    for i in range(1, len(vector) - 1):
        get_x = (vector[i] - vector[i - 1]) * (vector[i] - vector[i + 1])
        if (get_x >= threshold):
            slope_change += 1
    return slope_change


# New function
def getSSI(vector):
    vector = np.asarray(vector)
    return np.sum(np.square(vector))


# New function
def getSTD(vector):
    vector = np.asarray(vector)
    return np.std(vector)


def getTM(vector, order=3):
    vector = np.asarray(vector)
    return np.abs(np.mean(np.power(vector, order)))


# New function
def getTTP(vector):
    vector = np.asarray(vector)
    spec = np.fft.fft(vector)
    spec = spec[0:int(round(spec.shape[0] / 2))]
    POW = spec * np.conj(spec)
    return np.sum(POW)


# New function
def getVAR(vector):
    vector = np.asarray(vector)
    return np.square(np.std(vector))


# Ulysse function
def getWAMP(vector, threshold=0.1):
    vector = np.asarray(vector)
    wamp_decision = 0
    for i in range(1, len(vector)):
        get_x = abs(vector[i] - vector[i - 1])
        if (get_x >= threshold):
            wamp_decision += 1
    return wamp_decision


# New function
def getWL(vector):
    vector = np.asarray(vector)
    return np.sum(np.abs(np.diff(vector)))


# Ulysse function
def getZC(vector, threshold=0.1):
    vector = np.asarray(vector)
    number_zero_crossing = 0
    current_sign = cmp(vector[0], 0)
    for i in range(0, len(vector)):
        if current_sign == -1:
            if current_sign != cmp(vector[i], threshold):  # We give a delta to consider that the zero was crossed
                current_sign = cmp(vector[i], 0)
                number_zero_crossing += 1
        else:
            if current_sign != cmp(vector[i], -threshold):
                current_sign = cmp(vector[i], 0)
                number_zero_crossing += 1
    return number_zero_crossing


# Ulysse function
def getHjorth_activity_parameter(vector):
    return np.var(vector)


# Ulysse function
def getHjorth_mobility_parameter(vector):
    first_derivative = np.diff(vector)
    ratio = np.var(first_derivative, ddof=1) / np.var(vector, ddof=1)  # Sample variance
    return math.sqrt(ratio)


# Ulysse function
def getHjorth_complexity_parameter(vector):
    mobility_signal = getHjorth_mobility_parameter(vector)
    mobility_first_derivate = getHjorth_mobility_parameter(np.diff(vector))
    return mobility_first_derivate / mobility_signal


# Ulysse function
def mDWT_NinaPro_direct_implementation(vector, level=3, wavelet='db7'):
    coefficients = pywt.wavedec(vector, level=level, wavelet=wavelet)
    C = []
    for vector in coefficients:
        C.extend(vector)
    N = len(C)
    SMax = int(math.log(N, 2))
    Mxk = []
    for s in range(SMax):
        CMax = int(round((N / (2. ** (s + 1))) - 1))
        Mxk.append(np.sum(np.abs(C[0:(CMax)])))
    return Mxk


def getCoefficientVariation(vector):
    mean_vector = np.mean(vector)
    numerator = np.sqrt(np.sum(np.square(vector - mean_vector)) / (len(vector) - 1))
    denominator = np.sum(vector) / len(vector)
    return np.log(numerator / denominator)


def getTeagerKaiserEnergyOperator(vector):
    summation = 0.
    for i in range(1, len(vector) - 1):
        summation += vector[i] ** 2 - (vector[i - 1] * vector[i + 1])
    return summation


def getTDD(vector):
    lambda_variable = 0.1
    first_derivative = np.gradient(vector)
    second_derivative = np.gradient(first_derivative)
    moment_zero = np.power(getRootSquaredMoment(vector) / (len(vector) - 1), lambda_variable) / lambda_variable
    moment_two = np.power(getRootSquaredMoment(first_derivative) / (len(vector) - 1), lambda_variable) / lambda_variable
    moment_four = np.power(getRootSquaredMoment(second_derivative) / (len(vector) - 1),
                           lambda_variable) / lambda_variable
    sparseness = getSparseness(moment_zero=moment_zero, moment_two=moment_two, moment_four=moment_four)
    irregularity_factor = getIrregularityFactor(moment_zero=moment_zero, moment_two=moment_two, moment_four=moment_four)
    cov = getCoefficientVariation(vector)
    teager_kaiser_energy_operator_log = np.log(getTeagerKaiserEnergyOperator(vector))

    features_vector = [np.log(moment_zero), np.log(moment_zero - moment_two), np.log(moment_zero - moment_four),
                       sparseness, irregularity_factor, cov, teager_kaiser_energy_operator_log]
    return features_vector


def getTSD(vector_channels):
    final_feature_set = []
    for vector in vector_channels:
        features_vector_a = getTDD(vector)
        features_vector_b = getTDD(np.log(np.square(vector)))
        features_vector = getCosineSimilarityAsDescribedInFSD_article(features_vector_a, features_vector_b)
        final_feature_set.extend(features_vector)

    for i in range(len(vector_channels)):
        for j in range(i + 1, len(vector_channels)):
            vector_substracted = vector_channels[i] - vector_channels[j]
            features_vector_a = getTDD(vector_substracted)
            features_vector_b = getTDD(np.log(np.square(vector_substracted)))
            features_vector = getCosineSimilarityAsDescribedInFSD_article(features_vector_a, features_vector_b)
            final_feature_set.extend(features_vector)
    return final_feature_set


def getCosineSimilarityAsDescribedInFSD_article(vector_a, vector_b):
    resulting_orientation_vector = []
    for a_i, b_i in zip(vector_a, vector_b):
        f_i = (a_i * b_i) / (np.sqrt(np.sum(np.square(a_i))) + np.sqrt(np.sum(np.square(b_i))))
        resulting_orientation_vector.append(f_i)
    return resulting_orientation_vector


def getWL_second_order(first_derivative, second_derivative):
    return np.sum(np.abs(first_derivative)) / np.sum(np.abs(second_derivative))


def getRootSquaredMoment(vector):
    return np.sqrt(np.sum(np.square(vector)))


def getSparseness(moment_zero, moment_two, moment_four):
    denominator = np.sqrt(moment_zero - moment_two) * np.sqrt(moment_zero - moment_four)
    return np.log(moment_zero / denominator)


def getIrregularityFactor(moment_zero, moment_two, moment_four):
    return np.log(moment_two / (np.sqrt(moment_zero * moment_four)))


def getCosineSimilarityAsDescribedInOriginalTDPSD_article(vector_a, vector_b):
    resulting_orientation_vector = []
    for a_i, b_i in zip(vector_a, vector_b):
        f_i = (-2 * a_i * b_i) / (np.square(a_i) + np.square(b_i))
        resulting_orientation_vector.append(f_i)
    return resulting_orientation_vector


def getFeatures_for_tdpsd(vector):
    lambda_variable = 0.1
    first_derivative = np.gradient(vector)
    second_derivative = np.gradient(first_derivative)
    moment_zero = np.power(getRootSquaredMoment(vector), lambda_variable) / lambda_variable
    moment_two = np.power(getRootSquaredMoment(first_derivative), lambda_variable) / lambda_variable
    moment_four = np.power(getRootSquaredMoment(second_derivative), lambda_variable) / lambda_variable
    sparseness = getSparseness(moment_zero=moment_zero, moment_two=moment_two, moment_four=moment_four)
    irregularity_factor = getIrregularityFactor(moment_zero=moment_zero, moment_two=moment_two, moment_four=moment_four)
    features_vector = [np.log(moment_zero), np.log(moment_zero - moment_two), np.log(moment_zero - moment_four),
                       sparseness, irregularity_factor, np.log(getWL_second_order(first_derivative, second_derivative))]
    return features_vector


def get_TDPSD(vector):
    features_vector_a = getFeatures_for_tdpsd(vector)
    vector_log_scaled = np.log(np.square(vector))
    features_vector_b = getFeatures_for_tdpsd(vector_log_scaled)
    return np.array(getCosineSimilarityAsDescribedInOriginalTDPSD_article(features_vector_a, features_vector_b),
                    dtype=np.float32)


def get_LSF9(vector, threshold=1.0):
    features = [getL_scale(vector), getMFL(vector), getMSR(vector), getWAMP(vector, threshold=threshold),
                getZC(vector, threshold=threshold), getRMS(vector), getIEMG(vector), getDASDV(vector), getVAR(vector)]
    return np.array(features, dtype=np.float32)


def get_Sampen_pipeline_features_set(vector):
    features = getSampEn(vector)
    features.append(getRMS(vector=vector))
    features.append(getWL(vector))
    features.extend(getCC(vector, order=4))
    return np.array(features, dtype=np.float32)


def get_NinaPro_best(vector):
    features = get_TD_features_set(vector).tolist()
    features.append(getRMS(vector))
    features.extend(getHIST(vector, threshold_nmbr_of_sigma=3, bins=20))
    features.extend(mDWT_NinaPro_direct_implementation(vector, wavelet='db7', level=3))
    return np.array(features, dtype=np.float32)


def get_enhancedTD_feature_set(vector, threshold=1.):
    features = []
    features.extend(get_TD_features_set(vector=vector, threshold=threshold))
    features.append(getSKEW(vector))
    features.append(getRMS(vector))
    features.append(getIEMG(vector))
    features.extend(getAR(vector, 11))
    features.append(getHjorth_activity_parameter(vector))
    features.append(getHjorth_mobility_parameter(vector))
    features.append(getHjorth_complexity_parameter(vector))

    return np.array(features, dtype=np.float32)


def get_TD_features_set(vector, threshold=1.):
    features = [getMAV(vector), getZC(vector, threshold=threshold), getSSC(vector, threshold=threshold), getWL(vector)]
    return np.array(features, dtype=np.float32)


def get_dataset_with_features_set(dataset, features_set_function=get_TD_features_set):
    dataset_to_return = []
    for example in dataset:
        example_formatted = []
        all_zero_channel = False
        if features_set_function is getTSD:
            for vector_electrode in example:
                # If only 0. the sensor was not recording correctly and we should ignore this example
                if np.sum(vector_electrode) == 0:
                    all_zero_channel = True
            if all_zero_channel is False:
                example_formatted = features_set_function(example)
        else:
            for vector_electrode in example:
                # If only 0. the sensor was not recording correctly and we should ignore this example
                if np.sum(vector_electrode) != 0:
                    example_formatted.append(features_set_function(vector_electrode))
                else:
                    all_zero_channel = True
        if all_zero_channel is False:
            dataset_to_return.append(np.array(example_formatted).transpose().flatten())
    return dataset_to_return


if __name__ == '__main__':
    a = [4.0364, 9.3522, -7.4884, -29.285, -6.1338, 3.0227, 22.24, 56.407, 34.754, 40.978, 23.505, 9.8179, 35.561,
         14.979, 17.96, -16.515, -28.289, -40.667, 38.803, 22.503, 19.205, -19.043, -34.123, -24.346, -54.224, -28.449,
         -6.1375, -70.389, -36.899, -2.1967, 13.465, -0.89412, 7.9374, 3.5035, 52.504, 36.984, 4.1532, 25.537, 41.873,
         38.147, 13.645, -38.204, -102.55, -53.543, 33.253, 45.099, 8.0257, 10.686, 22.726, 52.178, 9.3122, -19.47,
         -4.2596, 12.7, 19.97, -1.3481, -24.034, -30.646, -23.297, -29.211, -27.832, -27.049, -38.635, -46.151, 11.309,
         -2.4908, 30.031, 52.965, 28.569, 38.263, 28.971, 17.451, 37.29, 51.586, 51.582, 26.725, -23.095, -13.071,
         -127.68, -72.734, -45.092, -55.192, -28.271, -51.371, -53.138, -31.197, -1.6199, 49.398, 101.34, 68.465,
         58.786, 35.03, 31.726, 43.093, 42.144, 1.6165, -40.013, -75.465, -43.826, 24.753, 72.596, 87.152, 54.129,
         14.602, -44.362, -68.031, -52.01, -77.875, -98.936, -81.045, -27.242, 74.372, 107, 85.314, 37.729, 1.7476,
         -3.0569, 12.673, -132.35, -74.879, -54.101, 5.1472, 44.755, -10.166, 10.312, 9.2823, 30.675, 41.601, 8.9585,
         -1.0946, 4.2738, -12.697, -2.2753, 5.8905, 8.3999, 11.748, 24.368, -63.069, -79.326, 12.473, 62.336, 65.376,
         61.344, 49.618, -2.3384, -13.851, -14.773, -39.104, -30.04, -17.228, -42.235]
    b = [-169.14, 35.79, 404.24, 853.98, 1272, 1819.5, 2289.8, 2599.2, 2293.9, 1765.7, 974.72, -822.49, -1378, -1028.8,
         -159.75, 1149.3, 1633.2, 1628.8, 1099.9, -1673.1, -1352, -833.11, -391.54, 240.37, 1185.8, 1146.9, 147.54,
         -1545.7, -1901.9, -1912.7, -1385.4, -799.15, 264.87, 918.46, 1199.1, -72.77, -1169.6, -1469, -1761.6, -1464.6,
         -730.7, -117.85, 409.76, 1669.5, 1808.3, 1718.4, 817.19, -1141.8, -1319.3, -1122.8, -23.83, 583.36, 1207.2,
         1903.1, 2226.9, 1260, 528.09, -1.6473, -549.28, -894.44, -1066.4, -939.12, -765.56, -354.39, 461.14, 1618.2,
         1812.8, 1521.5, -419.03, -867.18, -1319.1, -1507.7, -1495.5, -1232, -871.99, -328, 773.63, 1479.8, 551.91,
         -490.93, -896.28, -923.17, -499.76, 45.458, 1177, 1783.6, 2120.8, 1891.3, 35.742, -538.28, -1038.7, -1352.5,
         -1450.8, -1290.3, -319.44, 212.48, 789.03, 1250.8, 1391.6, 1005.9, 417.76, -262.59, -852.6, -1347.5, -1951.1,
         -1105.3, -538.15, 104.97, 735.39, 1193.3, 1290.7, 821.05, 206.97, -919.64, -1334.6, -1631.7, -1508.3, -777.65,
         1073.3, 2148.7, 1917, 774.68, -774.44, -1094.3, -1099, -764.75, -290.68, 673.75, 1293.4, 1744.6, 1902, 1716.3,
         1419.4, 770.38, 139.77, -479.92, -1797.9, -1769.7, -1374.6, 486.1, 900.99, 854.1, 672.77, 179.75, -1123.2,
         -2729.4, -2660.8, -2498.4, -1042.4, -458.06, 925.83]
    c = [13.578, 53.829, -65.68, -200.31, -156.64, -139.58, -137.16, -149.81, -198.99, -168.64, -118.72, 5.3937, 165.24,
         38.995, -94.329, 24.045, 20.555, -21.383, 43.058, 143.96, 71.508, -53.514, -58.626, -60.29, -88.772, -10.751,
         3.7171, 13.774, 3.5227, 12.969, 14.325, 18.502, 102.77, 107.98, 13.426, -21.051, 8.7797, 28.82, 76.196, 64.903,
         47.919, 13.381, -12.934, 22.309, 98.637, 65.633, -117.47, -63.797, 1.5475, 21.797, -152.55, -17.855, 168.95,
         247.96, 107.76, -141.05, -213.4, -237.12, -154.74, -102.47, -71.443, -40.231, -18.62, -12.484, 17.69, 36.76,
         73.499, 142.54, 25.863, -25.085, -36.134, 19.989, 127.61, 194.86, 146.18, 23.604, -119.34, -105.69, -32.869,
         -32.964, -28.362, -17.177, 0.21596, -10.303, -35.115, -34.329, 81.652, 113.74, 60.512, -105.09, -104.55,
         -22.854, -2.5456, -16.565, 58.494, 83.75, 55.548, -19.934, -53.41, -20.641, -34.402, -49.394, -34.447, 17.79,
         0.43814, -101.11, -46.762, 4.9029, -4.0583, 2.8058, 88.535, 346.59, 381, 170.23, -174.66, -334.27, -133.48,
         -119.91, -200.43, -92.701, -66.501, -6.5912, 45.353, 105.46, 136.19, 124.53, 69.078, -129.32, -67.908, 9.9804,
         39.303, 49.505, 28.771, 12.308, -30.453, -54.57, 6.6631, 31.894, 36.134, -62.292, -32.043, 5.857, 2.1077,
         -2.6899, 120.55, 84.02, -7.7783, -30.075, -103.94, -91.047, -85.004]


    true_MFL_a = 3.5994
    true_MFL_b = 4.968
    true_MFL_c = 3.9504

    print(getMFL(a), " VS TRUE: ", true_MFL_a)
    print(getMFL(b), " VS TRUE: ", true_MFL_b)
    print(getMFL(c), " VS TRUE: ", true_MFL_c)

    print("")

    true_BC_a = 1.4454
    true_BC_b = 1.5325
    true_BC_c = 1.4026

    print(getBC(a), " VS TRUE: ", true_BC_a)
    print(getBC(b), " VS TRUE: ", true_BC_b)
    print(getBC(c), " VS TRUE: ", true_BC_c)

    print(getSampEn(a))
    print(getSampEn(b))
    print(getSampEn(c))