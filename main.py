from flask import Flask,jsonify,request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

@app.route("/")
def home():
    return jsonify({"message": "Hello, Flask!"})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "running", "message": "Flask backend is operational!"})


#Dataset Upload Endpoint
@app.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["file"]
    file.save(f"./uploads/{file.filename}")
    return {"message": f"{file.filename} uploaded successfully!"}, 200

#Model Configuration Endpoint
@app.route("/configure", methods=["POST"])
def configure_model():
    config = request.json  # JSON payload from frontend
    return {"message": "Model configured", "config": config}, 200

#Training Endpoint
@app.route("/train", methods=["POST"])
def train_model():
    # Placeholder for training logic
    return {"message": "Training started"}, 200

#Model Export Endpoint
@app.route("/export", methods=["GET"])
def export_model():
    return {"message": "Model exported successfully!"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

