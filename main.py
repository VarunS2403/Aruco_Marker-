import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
params     = cv2.aruco.DetectorParameters()
detector   = cv2.aruco.ArucoDetector(aruco_dict, params)

THRESHOLD = 10
FRAME_W, FRAME_H = 640, 480

def get_direction(dx, dy, threshold):
    if abs(dx) < threshold and abs(dy) < threshold:
        return "LOCK ENGAGED", (0, 255, 0)

    angle = math.degrees(math.atan2(-dy, dx))

    if -22.5 <= angle < 22.5:
        return "MOVE LEFT", (0, 0, 255)
    elif 22.5 <= angle < 67.5:
        return "MOVE UP-LEFT", (255, 100, 0)
    elif 67.5 <= angle < 112.5:
        return "MOVE UP", (0, 0, 255)
    elif 112.5 <= angle < 157.5:
        return "MOVE UP-RIGHT", (255, 100, 0)
    elif angle >= 157.5 or angle < -157.5:
        return "MOVE RIGHT", (0, 0, 255)
    elif -157.5 <= angle < -112.5:
        return "MOVE DOWN-RIGHT", (255, 100, 0)
    elif -112.5 <= angle < -67.5:
        return "MOVE DOWN", (0, 0, 255)
    elif -67.5 <= angle < -22.5:
        return "MOVE DOWN-LEFT", (255, 100, 0)

    return "LOCK ENGAGED", (0, 255, 0)

def mirror_x(x):
    """Flip an x coordinate to match the mirrored display frame."""
    return FRAME_W - 1 - x

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (FRAME_W, FRAME_H))

    # --- Detect on the ORIGINAL (unflipped) frame ---
    corners, ids, _ = detector.detectMarkers(frame)

    # --- Now flip the frame for display ---
    display = cv2.flip(frame, 1)

    cx, cy = FRAME_W // 2, FRAME_H // 2

    # Draw crosshair at screen center (same in both frames since it's the center)
    cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 30, 2)

    if ids is not None:
        # Mirror the corners so they match the flipped display
        mirrored_corners = []
        for marker_corners in corners:
            mc = marker_corners.copy()
            mc[0][:, 0] = mirror_x(mc[0][:, 0])
            mirrored_corners.append(mc)

        # Draw marker outline on the mirrored display
        cv2.aruco.drawDetectedMarkers(display, mirrored_corners, ids)

        # Get centroid from ORIGINAL corners (for correct direction math)
        pts_original = corners[0][0]
        orig_marker_x = int(pts_original[:, 0].mean())
        orig_marker_y = int(pts_original[:, 1].mean())

        # Mirror the centroid for drawing on display
        display_marker_x = mirror_x(orig_marker_x)
        display_marker_y = orig_marker_y  # y doesn't change in horizontal flip

        # Error vector uses ORIGINAL coordinates (real-world directions)
        dx = orig_marker_x - cx
        dy = orig_marker_y - cy

        # Draw vector arrow on DISPLAY using mirrored coords
        cv2.arrowedLine(
            display,
            (cx, cy),
            (display_marker_x, display_marker_y),
            (0, 200, 255),
            2,
            tipLength=0.05
        )

        # Draw dot at mirrored centroid
        cv2.circle(display, (display_marker_x, display_marker_y), 6, (0, 200, 255), -1)

        # Direction command (calculated from original real-world coords)
        command, color = get_direction(dx, dy, THRESHOLD)

        # Display text on mirrored frame
        cv2.putText(display, command, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.putText(display, f"dx: {dx}  dy: {dy}", (20, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.putText(display, f"ID: {ids[0][0]}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        marker_width = int(np.linalg.norm(pts_original[0] - pts_original[1]))
        if command == "LOCK ENGAGED":
            if marker_width < 80:
                cv2.putText(display, "APPROACH", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            else:
                cv2.putText(display, "HOLD POSITION", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        cv2.putText(display, "SEARCHING...", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 128, 128), 2)

    cv2.imshow("ArUco Tracker", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    

    