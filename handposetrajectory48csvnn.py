import cv2
import mediapipe as mp
import csv

# MediaPipe Initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Webcam Initialization
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# MediaPipe Hands Initialization
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3) as hands:
    # CSV file initialization
    csv_file = open('hand_coordinates.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Finger', 'X', 'Y', 'Z'])

    frame_count = 0

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

            # List of finger landmarks and their corresponding names
            finger_landmarks = [
                (mp_hands.HandLandmark.THUMB_TIP, "Thumb"),
                (mp_hands.HandLandmark.INDEX_FINGER_TIP, "Index finger"),
                (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, "Middle finger"),
                (mp_hands.HandLandmark.RING_FINGER_TIP, "Ring finger"),
                (mp_hands.HandLandmark.PINKY_TIP, "Pinky finger")
            ]

            # Iterate through finger landmarks
            for landmark, finger_name in finger_landmarks:
                # Get the 3D coordinates of the finger tip
                finger_tip = hand_landmarks.landmark[landmark]

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Display the 3D coordinates of the finger tip
                print(f"{finger_name}: X={finger_tip.x}, Y={finger_tip.y}, Z={finger_tip.z}")

                # Write the coordinates to the CSV file
                csv_writer.writerow([frame_count, finger_name, finger_tip.x, finger_tip.y, finger_tip.z])

        # Display the frame
        cv2.imshow('Hand Pose Estimation', frame)

        # Increment frame count
        frame_count += 1

        # Check for key press and exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Close the CSV file
    csv_file.close()

