import cv2
import dlib
import numpy as np
import mediapipe as mp
import os
import time
from deepface import DeepFace
from flask import Flask, request, jsonify

# Create a folder to store reference faces
FACES_DB = "faces_db"
os.makedirs(FACES_DB, exist_ok=True)

# Face Detection
detector = dlib.get_frontal_face_detector()
def detect_faces(image_path, save_folder=FACES_DB):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    detected_faces = []
    for i, face in enumerate(faces):
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_img = img[y:y+h, x:x+w]
        face_filename = os.path.join(save_folder, f"face_{i}.jpg")
        cv2.imwrite(face_filename, face_img)
        detected_faces.append(face_filename)
    return detected_faces

# Face Recognition
def verify_identity(live_face, reference_faces):
    best_match = None
    best_distance = float("inf")
    for ref_face in reference_faces:
        try:
            result = DeepFace.verify(live_face, ref_face, model_name='Facenet')
            if result['verified'] and result['distance'] < best_distance:
                best_match = ref_face
                best_distance = result['distance']
                print(f"✅ Match Found: {best_match}, Distance: {best_distance:.2f}")
        except Exception as e:
            print(f"Error comparing {live_face} with {ref_face}: {str(e)}")
    
    if best_match:
        print("✅ Final Match Verified!")
    else:
        print("❌ No Match Found.")
    
    return best_match, best_distance

# Liveness Detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
def detect_liveness(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    return results.multi_face_landmarks is not None

# Flask API Setup
app = Flask(__name__)

@app.route('/verify', methods=['POST'])
def verify():
    data = request.json
    img1 = data['img1']
    img2 = data['img2']
    verified, distance = verify_identity(img1, [img2])
    return jsonify({"verified": bool(verified), "distance": distance})

# Live Webcam Face Verification
def live_face_verification():
    cap = cv2.VideoCapture(0)  # Open webcam
    start_time = time.time()
    detected = False
    
    while time.time() - start_time < 15:  # Run for 15 seconds
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite("live_face.jpg", face_img)
            
            # Verify the captured face with all reference images
            stored_faces = [os.path.join(FACES_DB, f) for f in os.listdir(FACES_DB)]
            best_match, distance = verify_identity("live_face.jpg", stored_faces)
            label = f"Match: {best_match}" if best_match else "No Match"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Live Face Verification", frame)
            if best_match:
                detected = True
                break
        
        cv2.imshow("Live Face Verification", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Run Face Detection Example
    faces = detect_faces("sample.jpg")
    if faces:
        print("Stored reference faces successfully.")
        print("Starting Live Face Verification...")
        live_face_verification()
    else:
        print("No reference face detected in sample.jpg.")
    
    # Run Flask Server
    app.run(debug=False)
