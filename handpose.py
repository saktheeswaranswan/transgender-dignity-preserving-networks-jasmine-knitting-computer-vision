import cv2
import mediapipe as mp

# MediaPipe Initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Webcam Initialization
cap = cv2.VideoCapture("handposenet.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# MediaPipe Hands Initialization
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5) as hands:
    finger_trajectory = []

    while cap.isOpened():
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirrored view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB and process it with MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        # Check if hand landmarks are detected
        if results.multi_hand_landmarks:
            # Extract hand landmarks for the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]

            # Get 3D coordinates of index finger and thumb finger
            index_finger_coords = (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x,
                                   hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y,
                                   hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z)
            thumb_finger_coords = (hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x,
                                   hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y,
                                   hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z)

            # Append finger coordinates to the trajectory list
            finger_trajectory.append((index_finger_coords, thumb_finger_coords))

            # Limit the trajectory list to a certain number of frames (e.g., 100)
            max_trajectory_length = 100
            if len(finger_trajectory) > max_trajectory_length:
                finger_trajectory = finger_trajectory[-max_trajectory_length:]

            # Draw hand landmarks on the frame
            annotated_image = frame.copy()
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Get pixel coordinates of the landmark
                height, width, _ = annotated_image.shape
                x, y = int(landmark.x * width), int(landmark.y * height)

                # Draw numbered nodes on the hand landmarks
                cv2.putText(annotated_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)

            # Display the annotated frame
            cv2.imshow('Hand Pose Estimation', annotated_image)
        else:
            # Display the frame without annotations if no hand landmarks are detected
            cv2.imshow('Hand Pose Estimation', frame)

        # Check for key press and exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

