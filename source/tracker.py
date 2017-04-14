import cv2
import numpy


class Tracker:
    def __init__(self, window_width, window_height, margin, vertical_meters_per_pixel=1, horizontal_meters_per_pixel=1,
                 smooth_factor=15):
        self.recent_centers = []
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.vertical_meters_per_pixel = vertical_meters_per_pixel
        self.horizontal_meters_per_pixel = horizontal_meters_per_pixel
        self.smooth_factor = smooth_factor

    def find_window_centroids(self, warped):
        window_centroids = []  # Store the (left,right) window centroid positions per level
        window = numpy.ones(self.window_width)  # Create our window template that we will use for convolutions

        # First find the two starting positions for the left and right lane by using numpy.sum to get the vertical image slice
        # and then numpy.convolve the vertical image slice with the window template 

        # Sum quarter bottom of image to get slice, could use a different ratio
        l_sum = numpy.sum(warped[int(3 * warped.shape[0] / 4):, :int(warped.shape[1] / 2)], axis=0)
        l_center = numpy.argmax(numpy.convolve(window, l_sum)) - self.window_width / 2
        r_sum = numpy.sum(warped[int(3 * warped.shape[0] / 4):, int(warped.shape[1] / 2):], axis=0)
        r_center = numpy.argmax(numpy.convolve(window, r_sum)) - self.window_width / 2 + int(warped.shape[1] / 2)

        # Add what we found for the first layer
        window_centroids.append((l_center, r_center))

        # Go through each layer looking for max pixel locations
        for level in range(1, (int)(warped.shape[0] / self.window_height)):
            # convolve the window into the vertical slice of the image
            image_layer = numpy.sum(warped[int(warped.shape[0] - (level + 1) * self.window_height):int(
                warped.shape[0] - level * self.window_height), :], axis=0)
            conv_signal = numpy.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use self.window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = self.window_width / 2
            l_min_index = int(max(l_center + offset - self.margin, 0))
            l_max_index = int(min(l_center + offset + self.margin, warped.shape[1]))
            l_center = numpy.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
            # Find the best right centroid by using past right center as a reference
            r_min_index = int(max(r_center + offset - self.margin, 0))
            r_max_index = int(min(r_center + offset + self.margin, warped.shape[1]))
            r_center = numpy.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
            # Add what we found for that layer
            window_centroids.append((l_center, r_center))
        self.recent_centers.append(window_centroids)
        return numpy.average(self.recent_centers[-self.smooth_factor:], axis=0)


def draw_windows(warped, window_centroids, window_width, window_height):
    l_points = numpy.zeros_like(warped)
    r_points = numpy.zeros_like(warped)

    left_x = []
    right_x = []
    # Go through each level and draw the windows 	
    for level in range(0, len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    # Draw the results
    template = numpy.array(r_points + l_points, numpy.uint8)  # add both left and right window pixels together
    zero_channel = numpy.zeros_like(template)  # create a zero color channle 
    template = numpy.array(cv2.merge((zero_channel, template, zero_channel)), numpy.uint8)  # make window pixels green
    warpage = numpy.array(cv2.merge((warped, warped, warped)),
                          numpy.uint8)  # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results
    return output, left_x, right_x


def window_mask(width, height, img_ref, center, level):
    output = numpy.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output
