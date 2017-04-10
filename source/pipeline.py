import cv2
import glob

from source.processor import process_image

files = glob.glob('../test_images/test*.jpg')

for index, file in enumerate(files):
    image = cv2.imread(file)
    processed_image = process_image(image)

    correction_result = str(index) + ' curve.jpg'
    cv2.imwrite('../output_images/' + correction_result, processed_image)
