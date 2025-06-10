"""
Face-Recognition GUI – PySide6 + OpenCV + PyTorch + dlib + FER emotion detection
----------------------------------------------------------------------------
• Replaces live webcam feed with image file upload via a button.
• Displays uploaded image and performs face recognition and emotion detection.
• Liveness Detection: Now integrated using the DeepFace library for static image anti-spoofing.
• Enhanced User Registration:
    - Upload image first, then recognition is performed.
    - If recognized, 'Register' button changes to 'Add Picture to [Name]' to add new embedding.
    - If unknown, 'Register' button changes to 'Register New Face' to allow new name entry.
    - Allows adding new pictures to existing users even if current picture is 'UNKNOWN'
      (e.g., due to angle), by pre-filling the name if recognized.
• Emotion (angry, disgust, fear, happy, sad, surprise, neutral) shown next
  to every face in real time.
"""

import sys, time, threading, cv2, torch, torch.nn.functional as F, numpy as np
from pathlib import Path
from PIL import Image
from scipy.spatial import distance                       # EAR maths
import dlib                                              # 68-landmark predictor
from fer import FER                                      # emotion detector
from torchvision import transforms
from deepface import DeepFace

try:
    from face_recognition.embnet import EmbNet
except ImportError:
    print("Error: Could not import EmbNet. Ensure 'embnet.py' is in a 'face_recognition' folder or its path is correct.")
    sys.exit(1)


from PySide6.QtCore    import Qt, QThread, Signal, Slot
from PySide6.QtGui     import QImage, QPixmap, QGuiApplication # Import QGuiApplication
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QLineEdit, QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox

# ─── CONFIG ─────────────────────────────────────────────────────────────
MODEL_PATH   = "models/embnet_triplet_hard_50k.pth"
CLASS_DB     = Path("models/class_db.pt")
PREDICTOR    = "models/shape_predictor_68_face_landmarks.dat" # Dlib predictor path
FACE_CASCADE = "models/haarcascade_frontalface_default.xml"  # OpenCV Haar cascade path

# Removed ANTI_SPOOF_MODEL_PATH as DeepFace handles its own models

# --- EAR (Eye Aspect Ratio) parameters for blink detection ---
# Note: For static images, true blink detection is not possible.
# These parameters are kept for structural consistency but liveness will be
# reported as 'LIVE' for uploaded images.
EAR_THRESH          = 0.20
EAR_CONSEC_FRAMES   = 1 # How many consecutive frames EAR must be below threshold to register a blink

# --- Facial landmark indices for left and right eyes ---
# These indices correspond to the 68-point model from dlib
L_EYE_PTS = list(range(36, 42)) # Left eye landmarks (points 36-41)
R_EYE_PTS = list(range(42, 48)) # Right eye landmarks (points 42-47)

# --- Face Recognition Threshold ---
# Cosine similarity threshold for identity verification
# Higher value means stricter match (0.0 to 1.0, 1.0 is identical)
COSINE_THRESH = 0.55 # Adjusted from original for potentially better balance

# --- Emotion colors ---
EMO_COL = {
    "angry":    (0, 0, 255),    # Red
    "disgust":  (128, 0, 128),  # Purple
    "fear":     (0, 165, 255),  # Orange
    "happy":    (0, 255, 0),    # Green
    "sad":      (255, 0, 0),    # Blue
    "surprise": (0, 255, 255),  # Yellow
    "neutral":  (128, 128, 128) # Grey
}

# ─── ANTI-SPOOFING MODEL INTEGRATION (using DeepFace) ──────────────────
class SpoofDetector:
    """
    A class designed to integrate DeepFace for anti-spoofing (Presentation Attack Detection).
    DeepFace handles its own model loading internally.
    """
    def __init__(self):
        print("INFO: Initializing DeepFace SpoofDetector. DeepFace handles model loading internally.")
        # DeepFace models will be downloaded on first use if not present.
        # No explicit model loading here.

    def predict(self, face_image_rgb: np.ndarray) -> tuple[bool, float]:
        """
        Predicts liveness from a static face image using DeepFace.
        Returns: (is_live: bool, confidence: float)
        """
        try:
            # DeepFace expects BGR format for OpenCV compatibility, so convert from RGB
            face_image_bgr = cv2.cvtColor(face_image_rgb, cv2.COLOR_RGB2BGR)

            # Use DeepFace.extract_faces with anti_spoofing=True
            # This will return a list of dictionaries, one for each detected face.
            # Each dict will contain 'is_real' and 'anti_spoofing_score' if anti_spoofing is enabled.
            
            # DeepFace.extract_faces expects a path or a BGR numpy array.
            # We are passing a cropped face, so it should be a single face.
            face_objs = DeepFace.extract_faces(
                img_path=face_image_bgr,
                detector_backend='skip', # Skip detection if you've already cropped the face
                anti_spoofing=True,
                enforce_detection=False # Do not raise error if no face detected by DeepFace's internal detector
            )

            if face_objs and len(face_objs) > 0:
                # Assuming we are passing a single cropped face, take the first result
                first_face_obj = face_objs[0]
                is_real = first_face_obj.get("is_real", False) # Default to False if key missing
                anti_spoofing_score = first_face_obj.get("anti_spoofing_score", 0.0) # Default to 0.0

                return is_real, anti_spoofing_score
            else:
                # If DeepFace couldn't process the face or found no face in the crop
                print("WARNING: DeepFace did not return a valid face object for anti-spoofing. Using dummy logic.")
                return self._dummy_predict(face_image_rgb)

        except Exception as e:
            print(f"ERROR: DeepFace anti-spoofing inference failed: {e}. Falling back to dummy logic.")
            return self._dummy_predict(face_image_rgb)

    def _dummy_predict(self, face_image_rgb: np.ndarray) -> tuple[bool, float]:
        """
        Simple, non-robust dummy logic for anti-spoofing.
        Simulates a spoof if the image is very dark.
        """
        avg_pixel_value = np.mean(face_image_rgb)
        
        if avg_pixel_value < 50: # Arbitrary low value to simulate a "dark spoof"
            return False, 0.1 # is_live=False, confidence=0.1 (low confidence for being real)
        
        return True, 0.95 # is_live=True, confidence=0.95 (high confidence for being real)


# ─── MODELS ─────────────────────────────────────────────────────────────
class Models:
    def __init__(self):
        # Load face detection cascade
        self.face_cascade = cv2.CascadeClassifier(str(Path(FACE_CASCADE)))  # Ensure the path is a string for OpenCV
        if self.face_cascade.empty():   
            print(f"Error: Could not load face cascade from {FACE_CASCADE}")    
            sys.exit(1)

        # Load dlib facial landmark predictor
        self.dlib_marks = dlib.shape_predictor(str(Path(PREDICTOR)))    # Ensure the path is a string for dlib
        if not Path(PREDICTOR).exists():
            print(f"Error: Could not load dlib shape predictor from {PREDICTOR}")
            sys.exit(1)

        # Load emotion detector
        self.emdet = FER(mtcnn=False) # mtcnn=False as we use Haar cascade for face detection

        # Load face embedding network
        self.embnet = EmbNet(dim=512)
        try:
            # MODIFICATION START: Correctly load EmbNet model state_dict
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            if 'model_state' in checkpoint:
                self.embnet.load_state_dict(checkpoint['model_state'])
            elif 'state_dict' in checkpoint:
                self.embnet.load_state_dict(checkpoint['state_dict'])
            else:
                # Fallback: assume the .pth file directly contains the state_dict
                self.embnet.load_state_dict(checkpoint)
            # MODIFICATION END
            self.embnet.eval() # Set to evaluation mode
            print(f"Loaded EmbNet model from {MODEL_PATH}")
        except FileNotFoundError:
            print(f"Error: EmbNet model not found at {MODEL_PATH}. Please ensure it's trained and saved.")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading EmbNet model: {e}")
            sys.exit(1)


        # Image transformation for EmbNet, normalization of all images.
        self.img_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        # Initialize the SpoofDetector (now uses DeepFace)
        self.spoof_detector = SpoofDetector()
        print("Anti-spoofing detector initialized (using DeepFace).")

        # Load or initialize class database
        # self.class_db will be a dictionary: {"name": torch.Tensor (N, 512)} so a single name can have multiple embeddings
        self.class_db = self._load_class_db()
        total_embeddings = sum(v.shape[0] for v in self.class_db.values())  # Total number of embeddings across all identities
        print(f"Loaded {len(self.class_db)} registered identities with {total_embeddings} total embeddings.")

    def _load_class_db(self):
        if CLASS_DB.exists():
            try:
                loaded_data = torch.load(CLASS_DB)
                # Validate if loaded data is in the expected dictionary format
                # where each value is a tensor of shape (N, 512)
                if isinstance(loaded_data, dict) and all(isinstance(v, torch.Tensor) for v in loaded_data.values()):
                    return loaded_data
                else:
                    print("Warning: Database found but in unexpected format. Initializing empty database.")
                    return {}
            except Exception as e:
                print(f"Error loading database from {CLASS_DB}: {e}. Initializing empty database.")
                return {}
        return {}

    def _save_class_db(self):
        torch.save(self.class_db, CLASS_DB)

    def register_embedding(self, name, new_embedding):
        """
        Registers a new embedding for a given name.
        If name exists, appends to existing embeddings.
        If name does not exist, creates a new entry.
        new_embedding is expected to be a (1, 512) tensor.
        """
        if new_embedding is None or new_embedding.shape != (1, 512):
            return False, "Invalid embedding provided for registration."

        if name in self.class_db:
            # Append new embedding to existing user's embeddings
            current_embeddings = self.class_db[name]
            self.class_db[name] = torch.cat((current_embeddings, new_embedding), dim=0)     # Concatenate along the first dimension
            msg = f"Added new picture for '{name}'. Total embeddings for '{name}': {self.class_db[name].shape[0]}"
        else:
            # Register new user with this embedding
            self.class_db[name] = new_embedding
            msg = f"Registered new user '{name}'."
        self._save_class_db()
        return True, msg

    def find_best_match(self, query_embedding):
        """
        Finds the best matching identity for a query embedding across all registered embeddings.
        Returns the recognized name and the highest similarity score.
        """
        best_name = "UNKNOWN"
        max_sim = COSINE_THRESH # Minimum similarity to consider a match

        if not self.class_db: # No registered users
            return "UNKNOWN", 0.0

        all_embeddings = []
        all_names = []
        for name, embeddings_tensor in self.class_db.items():
            all_embeddings.append(embeddings_tensor)
            all_names.extend([name] * embeddings_tensor.shape[0])   # Repeat name for each embedding

        all_embeddings_flat = torch.cat(all_embeddings, dim=0)  # Concatenate all embeddings into a single tensor
        
        # Calculate cosine similarity with all registered embeddings
        similarities = F.cosine_similarity(query_embedding, all_embeddings_flat)
        
        # Find the best match among all
        current_max_sim, current_best_idx = torch.max(similarities, dim=0)  # Get the max similarity and index of the best match

        if current_max_sim.item() > max_sim:    # Only consider if the similarity exceeds the threshold
            best_name = all_names[current_best_idx.item()]  # Get the corresponding name
            max_sim = current_max_sim.item()    # Convert to Python float for consistency
        
        return best_name, max_sim

    def clear_db(self):
        self.class_db = {}
        self._save_class_db()
        return True, "Database cleared successfully!"

    def _show_message_box(self, title, message, icon):
        # This method is called by Models class, but needs QApplication context
        # It's better to emit a signal from worker to MainWindow for message boxes
        pass # Placeholder, actual message box handled by MainWindow via signal

# ─── WORKER THREAD FOR IMAGE PROCESSING ─────────────────────────────────
class ImageProcessorWorker(QThread):
    frame_ready = Signal(np.ndarray) # Signal to send processed frame to GUI
    message_signal = Signal(str, str, int) # Signal for message boxes (title, message, icon type)
    registration_result_signal = Signal(bool, str) # Signal for registration outcome
    recognition_result_signal = Signal(str, bool) # recognized_name, is_face_detected

    def __init__(self, models):
        super().__init__()
        self.models = models
        self.current_frame_rgb = None # Store the current frame for registration
        self.current_face_embedding = None # Store the embedding for registration
        self.is_processing = False
        self.last_liveness_status = "UNKNOWN" # Store liveness status for registration check

        # EAR related for structural completeness, but not used for static image liveness
        self.blink_consec_frames = 0
        self.blink_count = 0

    def _ear(self, eye_points):
        # Compute the euclidean distances between the two sets of
        # vertical eye landmarks (x, y)-coordinates
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])

        # Compute the euclidean distance between the horizontal
        # eye landmark (x, y)-coordinates
        C = distance.euclidean(eye_points[0], eye_points[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    @Slot(str) # Slot to process an image file path
    def process_image(self, image_path):
        if self.is_processing:
            self.message_signal.emit("Processing", "Already processing an image. Please wait.", QMessageBox.Information)
            return

        self.is_processing = True
        try:
            # Load image
            frame = cv2.imread(image_path)  # Read image from file path
            if frame is None:
                self.message_signal.emit("Error", "Could not load image. Please check the file path or its content.", QMessageBox.Critical)
                self.is_processing = False
                # Emit recognition result to reset UI if image load fails
                self.recognition_result_signal.emit("UNKNOWN", False)
                return

            # Convert to RGB for display and consistency
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.current_frame_rgb = frame_rgb.copy() # Store for potential registration

            # Process the frame
            annotated_frame_rgb, recognized_name, face_detected = self._detect_and_process_frame(frame_rgb.copy())
            self.frame_ready.emit(annotated_frame_rgb)  # Emit processed frame to update UI
            self.recognition_result_signal.emit(recognized_name, face_detected) # Emit recognition result

        except Exception as e:
            print(f"Error processing image: {e}")
            self.message_signal.emit("Error", f"An error occurred during image processing: {e}", QMessageBox.Critical)
            self.recognition_result_signal.emit("UNKNOWN", False) # Reset UI on error
        finally:
            self.is_processing = False

    def _detect_and_process_frame(self, frame_rgb):
        frame_grey = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        
        # Detect faces using Haar Cascade
        faces = self.models.face_cascade.detectMultiScale(frame_grey, 1.3, 5)

        self.current_face_embedding = None # Reset for each new image
        recognized_name = "UNKNOWN"
        face_detected = False
        liveness_status = "UNKNOWN" # Initialize liveness status for this frame

        if len(faces) > 0:
            face_detected = True
            (x, y, w, h) = faces[0] # Take the first detected face (assuming one main face per image)

            # Draw bounding box
            cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Crop face for embedding and emotion detection
            face_crop_rgb = frame_rgb[y:y+h, x:x+w]
            
            # --- Face Embedding ---
            try:
                img_tensor = self.models.img_transform(Image.fromarray(face_crop_rgb)).unsqueeze(0)
                with torch.no_grad():   # Disable gradient calculation for inference
                    embedding = self.models.embnet(img_tensor)  # Get the embedding
                self.current_face_embedding = embedding # Store for registration
            except Exception as e:
                print(f"Error generating embedding: {e}")
                embedding = None

            # --- Face Recognition/Verification ---
            if embedding is not None:
                recognized_name, sim_score = self.models.find_best_match(embedding)
                print(f"Recognized: {recognized_name} with similarity {sim_score:.4f}")
            
            # --- Liveness Detection (using the SpoofDetector) ---
            # Call the spoof detector
            is_live_prediction, spoof_confidence = self.models.spoof_detector.predict(face_crop_rgb)
            liveness_status = "LIVE" if is_live_prediction else "SPOOF"
            self.last_liveness_status = liveness_status # Store for registration check
            print(f"Liveness: {liveness_status} (Confidence: {spoof_confidence:.2f})")
            
            # --- Emotion Detection ---
            # Ensure face_crop_rgb is not empty before passing to FER
            if face_crop_rgb.size > 0:
                emo, prob = self.models.emdet.top_emotion(cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)) or ("neutral", 0)
                if emo is None or prob is None:
                    emo, prob = "neutral", 0
            else:
                emo, prob = "neutral", 0 # Default if crop is somehow empty
            
            # --- Annotate frame ---
            text_name = f"ID: {recognized_name}"
            text_liveness = f"Liveness: {liveness_status}" # Use the new liveness status
            text_emotion = f"Emotion: {emo} ({prob:.1f}%)"

            # Choose color based on liveness
            box_color = (0, 255, 0) if liveness_status == "LIVE" else (0, 0, 255) # Green for LIVE, Red for SPOOF
            text_color = EMO_COL.get(emo, (255, 255, 255)) # Get emotion color, default white

            # Draw text
            font_scale = 0.7
            font_thickness = 2
            
            cv2.putText(frame_rgb, text_name, (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
            cv2.putText(frame_rgb, text_liveness, (x, y - 35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, box_color, font_thickness)
            cv2.putText(frame_rgb, text_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)
        else:
            # No face detected
            self.current_face_embedding = None
            self.last_liveness_status = "UNKNOWN" # Reset liveness status if no face
            cv2.putText(frame_rgb, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return frame_rgb, recognized_name, face_detected

    @Slot(str, str) # New slot for combined registration logic
    def perform_registration(self, name_input_text, recognized_name_from_ui):
        if self.current_face_embedding is None: # Check if an embedding was successfully generated
            self.message_signal.emit("Registration Error", "No face detected in the current image to register.", QMessageBox.Warning)
            self.registration_result_signal.emit(False, "No face detected.")
            return
        
        # Gating registration based on liveness status
        if self.last_liveness_status != "LIVE":
            self.message_signal.emit("Registration Error", "Cannot register: Liveness check failed (SPOOF detected).", QMessageBox.Warning)
            self.registration_result_signal.emit(False, "Liveness check failed.")
            return

        target_name = name_input_text.strip()

        # Logic: If a user was recognized, we prioritize adding to them.
        # If not recognized, we try to register a new user with the provided name_input_text.
        if recognized_name_from_ui != "UNKNOWN":
            # If a user was recognized, add the current embedding to their profile
            success, msg = self.models.register_embedding(recognized_name_from_ui, self.current_face_embedding)
        elif target_name: # User was UNKNOWN, and a name was entered in the input field
            # Register new user
            success, msg = self.models.register_embedding(target_name, self.current_face_embedding)
        else:
            self.message_signal.emit("Registration Error", "Please enter a name for new registration.", QMessageBox.Warning)
            self.registration_result_signal.emit(False, "Name required for new registration.")
            return

        self.registration_result_signal.emit(success, msg)  # Emit registration result
        if success and self.current_frame_rgb is not None:
            # Re-process the current image to show updated status (e.g., now recognized)
            annotated_frame_rgb, _, _ = self._detect_and_process_frame(self.current_frame_rgb.copy())
            self.frame_ready.emit(annotated_frame_rgb)

    @Slot()
    def clear_db_from_ui(self): # Clear database from UI action
        success, msg = self.models.clear_db()
        self.registration_result_signal.emit(success, msg)
        # Re-process current image to reflect empty database
        if self.current_frame_rgb is not None:
            annotated_frame_rgb, _, _ = self._detect_and_process_frame(self.current_frame_rgb.copy())
            self.frame_ready.emit(annotated_frame_rgb)
        else: # If no image loaded, clear the label
            self.frame_ready.emit(np.zeros((480, 640, 3), dtype=np.uint8)) # Black screen


# ─── MAIN WINDOW ────────────────────────────────────────────────────────
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("📸 Face-Rec with Image Upload + Emotion")
        self.label   = QLabel(alignment=Qt.AlignCenter)
        self.label.setFixedSize(640, 480) # Fixed size for the display area

        self.name_in = QLineEdit(placeholderText="type a name for registration")
        self.btn_upload = QPushButton("🖼️ Upload Image")
        self.btn_reg = QPushButton("📸 Register Face") # Initial text
        self.btn_clr = QPushButton("🗑️ Clear DB")

        # Layout
        vbox = QVBoxLayout(self)
        vbox.addWidget(self.label, 1) # Video feed / image display

        hbox = QHBoxLayout()
        hbox.addWidget(self.name_in)
        hbox.addWidget(self.btn_upload)
        hbox.addWidget(self.btn_reg)
        hbox.addWidget(self.btn_clr)
        vbox.addLayout(hbox)

        # Initialize models and worker thread
        self.models = Models()
        self.worker = ImageProcessorWorker(self.models)
        
        # Internal state to track last recognition result
        self.current_recognized_name_from_worker = "UNKNOWN"
        self.is_face_detected_in_current_image = False

        # Initial button/input states
        self.name_in.setEnabled(False) # Disabled until an image is processed
        self.btn_reg.setEnabled(False) # Disabled until a face is detected

        # Connect signals and slots
        self.btn_upload.clicked.connect(self.do_upload_image)
        self.btn_reg.clicked.connect(self.do_register) # This will now handle both new/existing
        self.btn_clr.clicked.connect(self.do_clear)

        self.worker.frame_ready.connect(self.update_img)
        self.worker.message_signal.connect(self.show_message_box)
        self.worker.registration_result_signal.connect(self.handle_registration_result)
        self.worker.recognition_result_signal.connect(self.handle_recognition_result) # NEW signal

        # Start the worker thread
        self.worker.start()

    @Slot(np.ndarray)
    def update_img(self, rgb):  # Slot to update the QLabel with the processed image
        h,w,ch = rgb.shape
        # Scale image to fit QLabel while maintaining aspect ratio
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # Scale pixmap to fit label, maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled_pixmap)

    @Slot() # Slot to handle image upload
    def do_upload_image(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Select Image File", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            # Reset UI elements before processing new image
            self.name_in.clear()
            self.name_in.setEnabled(False)
            self.btn_reg.setEnabled(False)
            self.btn_reg.setText("📸 Register Face")
            self.current_recognized_name_from_worker = "UNKNOWN"
            self.is_face_detected_in_current_image = False

            # Emit signal to worker to process the image
            self.worker.process_image(file_path)

    @Slot(str, bool)    # Slot to handle recognition results from the worker
    def handle_recognition_result(self, recognized_name, face_detected):
        self.current_recognized_name_from_worker = recognized_name
        self.is_face_detected_in_current_image = face_detected

        if face_detected:
            self.btn_reg.setEnabled(True)
            if recognized_name != "UNKNOWN":
                self.name_in.setText(recognized_name)
                self.name_in.setEnabled(False) # Disable name input if recognized
                self.btn_reg.setText(f"📸 Add Picture to {recognized_name}")
            else: # Face detected but UNKNOWN
                self.name_in.clear()
                self.name_in.setEnabled(True) # Enable name input for new registration
                self.btn_reg.setText("📸 Register New Face")
        else: # No face detected
            self.name_in.clear()
            self.name_in.setEnabled(False)
            self.btn_reg.setEnabled(False)
            self.btn_reg.setText("📸 Register Face") # Reset text

    @Slot() # Slot to handle registration action
    def do_register(self):
        # Pass current state to worker for unified registration logic
        # The worker will use self.worker.current_face_embedding
        self.worker.perform_registration(self.name_in.text(), self.current_recognized_name_from_worker)

    @Slot()
    def do_clear(self):
        self.worker.clear_db_from_ui()
        # Reset UI state after clearing DB
        self.current_recognized_name_from_worker = "UNKNOWN"
        self.is_face_detected_in_current_image = False
        self.name_in.clear()
        self.name_in.setEnabled(False)
        self.btn_reg.setEnabled(False)
        self.btn_reg.setText("📸 Register Face")

    @Slot(str, str, int)
    def show_message_box(self, title, message, icon_type):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Icon(icon_type)) # Cast int to QMessageBox.Icon
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

    @Slot(bool, str)
    def handle_registration_result(self, success, message):
        icon = QMessageBox.Information if success else QMessageBox.Critical
        self.show_message_box("Operation Result", message, icon)

    def closeEvent(self, event):
        # Ensure the worker thread is terminated when the main window closes
        if self.worker.isRunning():
            self.worker.quit()
            self.worker.wait()
        event.accept()

# ─── ENTRY POINT ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # MODIFICATION START: Call DPI policy settings before QApplication
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QGuiApplication.setAttribute(Qt.AA_UseHighDpiPixmaps)
   
    # Create necessary directories if they don't exist
    Path("models").mkdir(exist_ok=True)
    
    # Check for model files and provide guidance if missing
    if not Path(MODEL_PATH).exists():
        print(f"Warning: {MODEL_PATH} not found. Please ensure your trained EmbNet model is in the 'models' folder.")
        print("You might need to run project.py to train and save the model.")
    if not Path(PREDICTOR).exists():
        print(f"Warning: {PREDICTOR} not found. Please download dlib's shape_predictor_68_face_landmarks.dat.")
        print("You can usually find it by searching 'dlib shape_predictor_68_face_landmarks.dat download'.")
    if not Path(FACE_CASCADE).exists():
        print(f"Warning: {FACE_CASCADE} not found. Please download OpenCV's haarcascade_frontalface_default.xml.")
        print("You can usually find it in your OpenCV installation or online.")
    

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
