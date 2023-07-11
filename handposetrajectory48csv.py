import cv2
import mediapipe as mp
import csv

# MediaPipe Initialization
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Webcam Initialization
cap = cv2.VideoCapture("ddd.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# MediaPipe Hands Initialization
with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3) as hands:
    # CSV file initialization
    csv_file = open('hand_coordinates.csv', 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Frame', 'Node 4 - X', 'Node 4 - Y', 'Node 4 - Z', 'Node 8 - X', 'Node 8 - Y', 'Node 8 - Z'])

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

            # Get the 3D coordinates of the 4th and 8th finger nodes
            node_4 = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            node_8 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display the 3D coordinates of the nodes
            print(f"Node 4: X={node_4.x}, Y={node_4.y}, Z={node_4.z}")
            print(f"Node 8: X={node_8.x}, Y={node_8.y}, Z={node_8.z}")

            # Write the coordinates to the CSV file
            csv_writer.writerow([frame_count, node_4.x, node_4.y, node_4.z, node_8.x, node_8.y, node_8.z])

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

