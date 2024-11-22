from flask import Flask, request, render_template, jsonify
import os
from PIL import Image
from yolov5 import YOLOv5
import face_recognition

# Initialize the YOLOv5 model
model = YOLOv5(r"C:\Users\mudit\Downloads\yolov5x.pt")

app = Flask(__name__)

# Function to detect objects in an image
def detect_objects_in_image(image_path):
    image = Image.open(image_path)
    results = model.predict(image)
    detected_objects = []
    for result in results.pred[0]:
        class_id = int(result[5])  # Class ID
        confidence = result[4]  # Confidence score
        label = results.names[class_id]
        detected_objects.append((label, confidence))
    return detected_objects

# Function to match text to detected objects
def match_text_to_objects(text, detected_objects):
    text = text.lower()
    for label, confidence in detected_objects:
        if text in label.lower():
            return True
    return False

# Content-based image retrieval function
def content_based_image_retrieval_with_detection(text, image_path):
    detected_objects = detect_objects_in_image(image_path)
    return match_text_to_objects(text, detected_objects)

# Compare faces in a folder
def compare_faces_in_folder(reference_image_path, folder_path, text):
    reference_image = face_recognition.load_image_file(reference_image_path)
    reference_encoding = face_recognition.face_encodings(reference_image)

    if len(reference_encoding) == 0:
        return "No face detected in the reference image."

    matching_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            current_image_path = os.path.join(folder_path, filename)
            current_image = face_recognition.load_image_file(current_image_path)
            current_encodings = face_recognition.face_encodings(current_image)

            if len(current_encodings) > 0:
                match = face_recognition.compare_faces([reference_encoding[0]], current_encodings[0])
                if match[0]:
                    is_match = content_based_image_retrieval_with_detection(text, current_image_path)
                    if is_match:
                        matching_images.append(current_image_path)

    return matching_images

# Flask route for the home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get the text input
            text = request.form["text"]

            # Get the uploaded reference image
            reference_image = request.files["image1"]
            reference_image_path = os.path.join("uploads", reference_image.filename)
            reference_image.save(reference_image_path)

            # Get the folder path
            folder_path = request.form["folder_path"]

            # Check if folder path exists
            if not os.path.isdir(folder_path):
                return jsonify({"error": "Invalid folder path provided."}), 400

            # Perform face comparison and object detection
            matching_images = compare_faces_in_folder(reference_image_path, folder_path, text)

            # Return results
            if matching_images:
                return jsonify({"success": True, "matching_images": matching_images}), 200
            else:
                return jsonify({"success": False, "message": "No matches found."}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template("index.html")

# Run the Flask app
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)  # Ensure uploads directory exists
    app.run(debug=True)
