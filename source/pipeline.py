import cv2
import glob
import pickle

from source.thresholds import threshold_pipeline

distortion_pickle = pickle.load(open('./calibration_pickle.p', 'rb'))
cameraMatrix = distortion_pickle['cameraMatrix']
distCoeffs = distortion_pickle['distCoeffs']

files = glob.glob('./test_images/test*.jpg')

for index, file in enumerate(files):
    image = cv2.undistort(cv2.imread(file), cameraMatrix, distCoeffs, None, cameraMatrix)

    image = threshold_pipeline(image)

    correction_result = str(index) + ' undistorted.jpg'
    cv2.imwrite('./output_images/' + correction_result, image)
