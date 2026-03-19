# Mo ta file: Script demo nho de doc anh va hien thi nhanh bang OpenCV.
# Ghi chu: Comment tieng Viet duoc bo sung de de doc va bao tri.
import cv2

img = cv2.imread(r"/home/levietanh/OCR plate/img/GTEL_CAMERA_235_BIEN_SO_KM06_00-01-50.266__17A44094__or.jpeg")
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import numpy as np
a = np.array([[1, 2], [3, 4]])
a.shape