import cv2
import glob
import pickle

from source.curvature import draw_curve
from source.thresholds import threshold_pipeline
from source.tracker import Tracker, draw_windows
from source.transformer import transform

distortion_pickle = pickle.load(open('../calibration_pickle.p', 'rb'))
cameraMatrix = distortion_pickle['cameraMatrix']
distCoeffs = distortion_pickle['distCoeffs']

files = glob.glob('../test_images/test*.jpg')

for index, file in enumerate(files):
    image = cv2.undistort(cv2.imread(file), cameraMatrix, distCoeffs, None, cameraMatrix)

    processed_image = threshold_pipeline(image)
    warped, m_inverse = transform(image, processed_image)

    window_width = 25
    window_height = 80

    curve_centers = Tracker(window_width, window_height, 25, 10 / 720, 4 / 384)
    window_centroids = curve_centers.find_window_centroids(warped)
    sliding_windows, left_x, right_x = draw_windows(warped, window_centroids, window_width, window_height)
    weighted_road = draw_curve(warped, image, left_x, right_x, m_inverse, curve_centers)

    correction_result = str(index) + ' undistorted.jpg'
    cv2.imwrite('../output_images/' + correction_result, weighted_road)
