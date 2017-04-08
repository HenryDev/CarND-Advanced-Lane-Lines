import pickle
import cv2

from source.curvature import draw_curve
from source.thresholds import threshold_pipeline
from source.tracker import Tracker, draw_windows
from source.transformer import transform


def process_image(image):
    distortion_pickle = pickle.load(open('../calibration_pickle.p', 'rb'))
    camera_matrix = distortion_pickle['cameraMatrix']
    dist_coeffs = distortion_pickle['distCoeffs']

    image = cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)
    processed_image = threshold_pipeline(image)
    warped, m_inverse = transform(image, processed_image)
    window_width = 25
    window_height = 80
    curve_centers = Tracker(window_width, window_height, 25, 10 / 720, 4 / 384)
    window_centroids = curve_centers.find_window_centroids(warped)
    sliding_windows, left_x, right_x = draw_windows(warped, window_centroids, window_width, window_height)
    weighted_road = draw_curve(warped, image, left_x, right_x, m_inverse, curve_centers)
    return weighted_road
