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
    warped, m = transform(image, processed_image)

    box_width = 25
    box_height = 80
    curve_centers = Tracker(box_width, box_height, 25, 10 / 720, 4 / 384)
    window_centroids = curve_centers.find_window_centroids(warped)
    sliding_windows, left_x, right_x = draw_windows(warped, window_centroids, box_width, box_height)
    road_curve = draw_curve(image, warped, box_width, box_height, left_x, right_x)

    correction_result = str(index) + ' undistorted.jpg'
    cv2.imwrite('../output_images/' + correction_result, road_curve)
