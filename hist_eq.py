import numpy as np

def histogram(image):
    # Calculate normalized histogram of an image.
    m, n = image.shape
    h = [0.0] * 1000
    for i in range(m):
        for j in range(n):
            h[image[i, j]] += 1
    return np.array(h)/(m * n)

def cum_sum(histogram):
    # Find cumulative sum of a numpy array, list.
    return [sum(histogram[:i + 1]) for i in range(len(histogram))]

def histogram_equalization(image):
    # Calculate histogram.
    h       = histogram(image)
    cdf     = np.array(cum_sum(h))
    sk      = np.uint8(255 * cdf)
    s1, s2  = image.shape
    y       = np.zeros_like(image)
    
    for i in range(0, s1):
        for j in range(0, s2):
            y[i, j] = sk[image[i, j]]
    H = histogram(y)

    # Return transformed image, original and new histogram
    # and transform function.
    return y, h, H, sk

