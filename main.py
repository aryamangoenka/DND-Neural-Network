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
    try:
        # Debugging: Log the raw request
        print("Raw Request Data:", request.data)
        print("Headers:", request.headers)

        # Parse the received JSON data
        data = request.get_json()
        print("Parsed JSON Data:", data)

        nodes = data.get('nodes')
        edges = data.get('edges')

        # Validate the input structure
        if not nodes or not edges:
            return jsonify({'error': 'Invalid configuration. Nodes and edges are required.'}), 400

        # Simulate training logic
        print("Received Nodes:", nodes)
        print("Received Edges:", edges)

        # Send a success response
        return jsonify({'message': 'Training started successfully!', 'status': 'success'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


#Model Export Endpoint
@app.route("/export", methods=["GET"])
def export_model():
    return {"message": "Model exported successfully!"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

