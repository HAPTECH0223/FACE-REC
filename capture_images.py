import cv2
import os
import time

# Create directory for captured images
if not os.path.exists('training_images'):
    os.makedirs('training_images')

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Capture parameters
image_count = 0
target_images = 100

print("=" * 60)
print("Face Image Capture (Step 1 of 2)")
print("=" * 60)
print(f"Target: {target_images} images")
print("\nInstructions:")
print("- Position your face clearly in front of camera")
print("- Move head slightly between captures")
print("- Try different angles and expressions")
print("- Ensure good lighting")
print("- Press 'q' to quit early")
print("\nStarting in 3 seconds...")
print("=" * 60)

time.sleep(3)

# Auto-capture settings
last_capture_time = time.time()
capture_interval = 0.5  # Capture every 0.5 seconds

while image_count < target_images:
    ret, frame = cap.read()
    
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect face
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Visualize detection
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
            
            # Show eyes for visual feedback
            face_roi_gray = gray[fy:fy+fh, fx:fx+fw]
            face_roi_color = frame[fy:fy+fh, fx:fx+fw]
            
            eyes = eye_cascade.detectMultiScale(face_roi_gray, 1.1, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(face_roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        # Display progress
        cv2.putText(frame, f"Captured: {image_count}/{target_images}", 
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Auto-capture when face is detected
        current_time = time.time()
        if len(faces) > 0 and (current_time - last_capture_time) >= capture_interval:
            # Save image
            image_filename = f"training_images/face_{image_count:04d}.jpg"
            cv2.imwrite(image_filename, frame)
            
            image_count += 1
            last_capture_time = current_time
            
            print(f"✓ Captured: {image_filename} ({image_count}/{target_images})")
            
            # Visual feedback
            cv2.putText(frame, "CAPTURED!", (200, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imshow("Capture Training Images", frame)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("\nCapture interrupted by user")
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("Image Capture Complete!")
print("=" * 60)
print(f"Total images captured: {image_count}")
print(f"Images saved in: training_images/")
print("\nNext step: Run 'python train_landmarks.py' to extract landmarks")
print("=" * 60)
