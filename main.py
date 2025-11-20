# These are libraries (tools) we need to import to make our program work

# 'threading' allows us to run multiple tasks at the same time without freezing
import threading

# 'cv2' is OpenCV - a library for working with images and video from webcams
import cv2

# 'numpy' is a math library (we import it but don't use it much in this code)
import numpy as np

# Load pre-trained face cascade classifiers
# These are pre-made patterns that help detect faces in images
# 'haarcascade_frontalface_default.xml' is a file that knows what faces look like
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Similar to above, but this one recognizes eyes in a face
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start capturing video from your webcam (camera 0 is the default camera)
# cv2.CAP_DSHOW is a Windows-specific setting that helps avoid lag
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# Set the width and height of the video capture to 640x480 pixels
# This makes it easier to process and faster
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# This counter helps us track how many frames we've processed
# We use it to check faces every 10 frames (to save processing power)
counter = 0

# This variable stores whether we found a matching face or not (True = found, False = not found)
face_match = False

# This list will store the faces from the reference image we're comparing against
reference_faces = []

# Try to load the reference image - this is the image we want to compare against
try:
    # Read the image file called 'reference.jpg' from the same folder as this program
    reference_img = cv2.imread("reference.jpg")
    
    # Check if the image was successfully loaded
    if reference_img is not None:
        # Convert the image to grayscale (black and white) - this helps face detection work better
        gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        
        # Detect all faces in the reference image
        # The numbers 1.3 and 5 are settings that control how sensitive the detection is
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # For each face found, extract it and store it in our list
        for (x, y, w, h) in faces:
            # x, y is the position, w, h is the width and height
            # We're cropping the face region from the image
            reference_faces.append(reference_img[y:y+h, x:x+w])
        
        # Tell the user how many faces we found
        print(f"Loaded {len(reference_faces)} reference face(s)")
    else:
        # If the file doesn't exist or can't be read, tell the user
        print("reference.jpg not found. Please place reference.jpg in the same directory")
        
except Exception as e:
    # If something goes wrong, print the error message
    print(f"Error loading reference: {e}")

# This is a function (a reusable block of code) that checks if a face matches
# 'frame' is the video image we're checking
def check_face(frame):
    # Tell Python we're using the 'face_match' variable from outside this function
    global face_match
    
    try:
        # Convert the frame to grayscale (black and white)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect all faces in the current frame
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
        # If we found faces in both the current frame AND the reference image
        if len(faces) > 0 and len(reference_faces) > 0:
            # Consider it a match (in real apps, you'd do more complex comparison here)
            face_match = True
        else:
            # No match found
            face_match = False
            
    except Exception as e:
        # If something goes wrong, print the error
        print(f"Error in face detection: {e}")
        face_match = False


# This is the main loop - it keeps running forever until we press 'q'
while True:
    # Read one frame (image) from the webcam
    # 'ret' tells us if we successfully got a frame (True/False)
    # 'frame' is the actual image
    ret, frame = cap.read()
    
    # Check if we successfully captured a frame
    if ret:
        # Only check for faces every 10 frames (to save processing power)
        if counter % 10 == 0:
            try:
                # Start a new thread (parallel task) to check for faces
                # This way, the video keeps playing smoothly while we process the face detection
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                pass
        
        # Add 1 to our counter
        counter += 1
        
        # Convert the frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect all faces in the current frame
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
        # For each face found in the frame
        for (x, y, w, h) in faces:
            # Draw a blue rectangle around the face
            # (x,y) is top-left corner, (x+w, y+h) is bottom-right corner
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract just the face region for eye detection
            roi_gray = gray_frame[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            # Draw a green rectangle around each eye found
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Display the match status at the bottom of the video
        if face_match and len(reference_faces) > 0:
            # If we found a match, show "Match!" in green text
            cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            # Otherwise show "No Match" in red text
            cv2.putText(frame, "No Match", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # Display the frame in a window called "Face Recognition"
        cv2.imshow("Face Recognition", frame)
    
    # Wait 1 millisecond for a key press
    key = cv2.waitKey(1)
    
    # If the user presses 'q', exit the loop and end the program
    if key == ord('q'):
        break

# Close all open windows when the program ends
cv2.destroyAllWindows()