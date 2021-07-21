from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2

image = 'images/example.jpeg'
model = 'model/shape_predictor_68_face_landmarks.dat'

def eye_aspect_ratio(eye):
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model)

image = cv2.imread(image)
image = imutils.resize(image, width=1600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

rects = detector(gray, 1)

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(rect)

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    leftEyeHull = cv2.convexHull(leftEye)
    rightEyeHull = cv2.convexHull(rightEye)
    cv2.drawContours(image, [leftEyeHull], -1, (255, 255, 255), -1)
    cv2.drawContours(image, [rightEyeHull], -1, (255, 255, 255), -1)

    cv2.putText(image, "nguyentrieuphong.com no.{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for (x, y) in shape:
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

cv2.imwrite('output.jpg', image)
cv2.imshow("Output", image)
cv2.waitKey(0)

