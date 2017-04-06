import cv2
import numpy


class Tracker:
    def __init__(self, box_width, box_height, margin, vertical_meters_per_pixel=1, horizontal_meters_per_pixel=1,
                 smooth_factor=15):
        self.recent_centers = []
        self.box_width = box_width
        self.box_height = box_height
        self.margin = margin
        self.vertical_meters_per_pixel = vertical_meters_per_pixel
        self.horizontal_meters_per_pixel = horizontal_meters_per_pixel
        self.smooth_factor = smooth_factor

    def find_window_centroids(self, warped):
        window_centroids = []
        window = numpy.ones(self.box_width)

        l_sum = numpy.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = numpy.argmax(numpy.convolve(window, l_sum)) - self.box_width / 2
        r_sum = numpy.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = numpy.argmax(numpy.convolve(window, r_sum)) - self.box_width / 2 + int(warped.shape[1] / 2)

        window_centroids.append((l_center, r_center))

        for level in range(1, int(warped.shape[0] / self.box_height)):
            vertical_slice = warped[int(warped.shape[0] - (level + 1) * self.box_height):int(
                warped.shape[0] - level * self.box_height), :]
            image_layer = numpy.sum(vertical_slice, axis=0)
            conv_signal = numpy.convolve(window, image_layer)

            offset = self.box_width / 2
            l_min_index = int(max(l_center + offset - self.margin, 0))
            l_max_index = int(min(l_center + offset + self.margin, warped.shape[1]))
            l_center = numpy.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset

            r_min_index = int(max(r_center + offset - self.margin, 0))
            r_max_index = int(min(r_center + offset + self.margin, warped.shape[1]))
            r_center = numpy.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            window_centroids.append((l_center, r_center))

        self.recent_centers.append(window_centroids)
        return numpy.average(self.recent_centers[-self.smooth_factor:], axis=0)


def draw_windows(warped, window_centroids, window_width, window_height):
    l_points = numpy.zeros_like(warped)
    r_points = numpy.zeros_like(warped)

    left_x = []
    right_x = []
    for level in range(0, len(window_centroids)):
        left_x.append(window_centroids[level][0])
        right_x.append(window_centroids[level][1])

        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        l_points[(l_points == 255) | (l_mask == 1)] = 255
        r_points[(r_points == 255) | (r_mask == 1)] = 255

    template = numpy.array(r_points + l_points, numpy.uint8)
    zero_channel = numpy.zeros_like(template)
    template = numpy.array(cv2.merge((zero_channel, template, zero_channel)), numpy.uint8)
    warpage = numpy.array(cv2.merge((warped, warped, warped)), numpy.uint8)
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)
    return output, left_x, right_x


def window_mask(width, height, img_ref, center, level):
    output = numpy.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output
