import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh

# Mouth landmarks
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Eyelid landmarks (most responsive to smiling)
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Eye corners (normalization)
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263

def dist(a, b):
    return math.dist(a, b)

cap = cv2.VideoCapture(0)

neutral_mouth = None
neutral_eye_height = None

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        smile_percentage = 0

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]

            def pt(i):
                lm = face.landmark[i]
                return int(lm.x * w), int(lm.y * h)

            # Mouth
            ml = pt(MOUTH_LEFT)
            mr = pt(MOUTH_RIGHT)

            # Eyes
            left_top = pt(LEFT_EYE_TOP)
            left_bottom = pt(LEFT_EYE_BOTTOM)
            right_top = pt(RIGHT_EYE_TOP)
            right_bottom = pt(RIGHT_EYE_BOTTOM)

            # Eye corners for normalization
            lc = pt(LEFT_EYE_CORNER)
            rc = pt(RIGHT_EYE_CORNER)

            # Distances
            mouth_width = dist(ml, mr)
            eye_distance = dist(lc, rc)

            left_eye_h = dist(left_top, left_bottom)
            right_eye_h = dist(right_top, right_bottom)
            eye_height = (left_eye_h + right_eye_h) / 2

            # Normalize
            mouth_index = (mouth_width / eye_distance) * 100
            eye_index = (eye_height / eye_distance) * 100

            # Auto-calibration
            if neutral_mouth is None:
                neutral_mouth = mouth_index
            if neutral_eye_height is None:
                neutral_eye_height = eye_index

            # Changes
            mouth_change = mouth_index - neutral_mouth     # increases when smiling
            eye_change = neutral_eye_height - eye_index    # increases when eyes shrink

            # Weight both signals
            smile_score = (mouth_change * 2.0) + (eye_change * 4.0)

            # Clamp to 0–100
            smile_percentage = int(max(0, min(100, smile_score)))

            # Debug dots (optional)
            cv2.circle(frame, ml, 3, (0, 255, 0), -1)
            cv2.circle(frame, mr, 3, (0, 255, 0), -1)

        cv2.putText(frame, f"Smile: {smile_percentage}%", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Smile Detector (Eyes + Mouth)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()


