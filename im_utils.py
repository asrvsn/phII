import numpy as np

def combine_grayscale(arr1: np.ndarray, arr2: np.ndarray):
    arr1 = (arr1 - arr1.min()) / (arr1.max() - arr1.min())
    arr2 = (arr2 - arr2.min()) / (arr2.max() - arr2.min())
    return np.maximum(arr1, arr2)

def combine_rgb(arr1: np.ndarray, arr2: np.ndarray):
    '''
    arr1: green
    arr2: purple
    like in Fiji
    '''
    # Create RGB image G/P
    rgb_image = np.stack((arr2, arr1, arr2), axis=-1).astype(float)
    # Normalize each channel
    for i in range(3):
        rgb_image[..., i] = (rgb_image[..., i] - rgb_image[..., i].min()) / (rgb_image[..., i].max() - rgb_image[..., i].min())
    return rgb_image