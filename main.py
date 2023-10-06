import cv2
import os
import numpy as np
import getpass

# Create the "images" folder if it doesn't exist
if not os.path.exists('images'):
    os.makedirs('images')

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('face.xml')

# Create a list to store detected faces in the current session
detected_faces_in_session = []

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face image
        face_image = frame[y:y + h, x:x + w]

        # Convert it to grayscale
        face_image_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

        # Resize the face image to a consistent size
        face_image_gray_resized = cv2.resize(face_image_gray, (100, 100))  # Adjust the size as needed

        # Get the computer username
        username = getpass.getuser()

        # Display the username inside the rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
        cv2.putText(frame, username, (x, y + h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Check if this face has already been detected in this session
        if all(np.array_equal(face_image_gray_resized, saved_face) for saved_face in detected_faces_in_session):
            # If it's a new face in this session, save it
            detected_faces_in_session.append(face_image_gray_resized)
            # Add the username to the file name when saving
            cv2.imwrite(f'images/face_{len(detected_faces_in_session)}_{username}.jpg', face_image)

    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
