import numpy
import cv2
import glob
import pickle

chessboard_points = numpy.zeros((6 * 9, 3), numpy.float32)
chessboard_points[:, :2] = numpy.mgrid[0:9, 0:6].T.reshape(-1, 2)

object_points = []
image_points = []

files = glob.glob('../camera_cal/calibration*.jpg')

for index, file in enumerate(files):
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    were_corners_found, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    if were_corners_found:
        object_points.append(chessboard_points)
        image_points.append(corners)

        cv2.drawChessboardCorners(image, (9, 6), corners, were_corners_found)
        result = str(index) + ' with corners drawn.jpg'
        cv2.imwrite('../output_images/' + result, image)

test_image = cv2.imread('../camera_cal/calibration1.jpg')
image_size = (test_image.shape[1], test_image.shape[0])
were_corners_found, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points,
                                                                                 image_size, None, None)
distortion_pickle = {'cameraMatrix': cameraMatrix, 'distCoeffs': distCoeffs}
pickle.dump(distortion_pickle, open('../calibration_pickle.p', 'wb'))
