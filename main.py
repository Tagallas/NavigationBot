import cv2
import numpy as np

cap = cv2.VideoCapture(1)  # 0 - pierwsza kamera podłączona do komputera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # shape detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Szukamy kwadratów/przeszkód
            cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
    
    cv2.imshow("Detected Shapes", frame)

    # # HSV (color) detection - nie działa
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # lower = (100, 50, 50)  # Przykładowe wartości dla niebieskiego
    # upper = (130, 255, 255)
    # mask = cv2.inRange(hsv, lower, upper)

    # cv2.imshow("Mask", mask)

    # # 3D to 2D transformation
    # pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])  # Punkty na obrazie
    # pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # Równy obraz

    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # warped = cv2.warpPerspective(frame, matrix, (width, height))


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()