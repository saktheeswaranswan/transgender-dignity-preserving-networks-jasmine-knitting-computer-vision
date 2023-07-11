import cv2
import mediapipe as mp
import csv

# MediaPipe Initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Webcam Initialization
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# MediaPipe Hands Initialization
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3) as hands:
    # CSV file initialization
    csv_file = open('hand_coordinates.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Node Number', 'X', 'Y', 'Z'])

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

            # Draw landmarks on the frame with numbering
            annotated_image = frame.copy()
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Get the coordinates of the landmark
                landmark_x = int(landmark.x * frame.shape[1])
                landmark_y = int(landmark.y * frame.shape[0])

                # Draw a red circle at the landmark position
                cv2.circle(annotated_image, (landmark_x, landmark_y), 5, (0, 0, 255), -1)

                # Display the landmark number
                cv2.putText(annotated_image, str(idx), (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Write the coordinates to the CSV file
                csv_writer.writerow([frame_count, idx, landmark.x, landmark.y, landmark.z])

            # Display the annotated image
            cv2.imshow('Hand Pose Estimation', annotated_image)

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

