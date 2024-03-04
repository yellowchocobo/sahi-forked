import numpy as np
import skimage
def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=0):
    """Converts a binary mask to polygon representation
    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.
    """
    polygon = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = skimage.measure.find_contours(padded_binary_mask, 0.5)

    # yolo can produce a mask where pixels are not interconnected
    # in this case the following line does not work
    contours = np.subtract(contours, 1)  # np.array(contours)

    for contour in contours:
        contour = close_contour(contour)
        contour = skimage.measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour
        # after padding and subtracting 1 we may get -0.5 points in our segmentation
        segmentation = [np.clip(i, 0.0, i).tolist() for i in segmentation]
        polygon.append(segmentation)

    return polygon