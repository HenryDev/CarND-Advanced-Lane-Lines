import cv2
import numpy


def transform(image, processed_image):
    image_size = (image.shape[1], image.shape[0])
    offset = image_size[0] * 0.25
    bottom_width = 0.76
    middle_width = 0.08
    height_percentage = 0.62
    bottom_trim = 0.935
    source = numpy.float32([[image_size[0] * (0.5 - middle_width / 2), image_size[1] * height_percentage],
                            [image_size[0] * (0.5 + middle_width / 2), image_size[1] * height_percentage],
                            [image_size[0] * (0.5 + bottom_width / 2), image_size[1] * bottom_trim],
                            [image_size[0] * (0.5 - bottom_width / 2), image_size[1] * bottom_trim]])
    destination = numpy.float32([[offset, 0],
                                 [image_size[0] - offset, 0],
                                 [image_size[0] - offset, image_size[1]],
                                 [offset, image_size[1]]])
    m = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(processed_image, m, image_size, flags=cv2.INTER_LINEAR)
    return warped, m
