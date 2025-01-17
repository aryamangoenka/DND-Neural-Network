# Drag-and-Drop Neural Network Builder

A web-based tool that allows users to visually design, train, and export custom neural networks using an intuitive drag-and-drop interface. This project is designed for users ranging from machine learning beginners to advanced practitioners, making neural network development more accessible and interactive.

## 🚀 Key Features

- **Drag-and-Drop Interface:** Build custom neural networks by dragging and connecting different types of layers.
- **Predefined Templates:** Quickly start with ready-made neural network templates for common tasks.
- **Real-Time Training Visualization:** Monitor training progress with real-time graphs and logs.
- **Export Options:** Export trained models in Python Script, Jupyter Notebook, TensorFlow SavedModel and Keras Model.
- **Customizable Parameters:** Edit layer-specific parameters (neurons, activation functions, dropout rate, etc.).

## 🎯 Target Audience

- Machine learning beginners and enthusiasts.
- Data scientists and researchers needing a quick model prototyping tool.
- Developers interested in neural network visualization tools.

## 💡 Motivation

Developing neural networks can be complex and time-consuming, especially for beginners. This project simplifies that process by offering an interactive visual builder and eliminating the need for manual coding during model design.

---

## 🛠 Tech Stack

### **Frontend:**
- **React** with **TypeScript** for the interactive UI
- **React Flow** for the drag-and-drop interface
- **SCSS** for custom styling
- **Bootstrap** for responsive design
- **Chart.js (react-chartjs-2)** for visualizing training progress
- **Axios** for HTTP requests
- **Socket.IO Client** for real-time updates

### **Backend:**
- **Flask** for REST API development
- **Flask-SocketIO** for real-time communication
- **TensorFlow** and **Keras** for building and training neural networks
- **Scikit-learn** for evaluation metrics (confusion matrix, regression metrics)
- **Pandas** for data handling
- **Matplotlib** and **Seaborn** for plotting metrics

### **Tools:**
- **Postman** for API testing
- **GitHub** for version control and collaboration

---

## ⚙️ Installation Guide

### 1. **Prerequisites**

- **Python:** 3.12.4 or later
- **Node.js:** Latest LTS version (18.x or later)
- **npm:** Latest version (comes with Node.js)
- **pip:** Python package installer

### 2. **Backend Setup (Flask)**

```bash
# Clone the repository
git clone https://github.com/aryamangoenka/dnd-neural-network.git
cd dnd-neural-network

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install backend dependencies
pip install -r requirements.txt

# Run the Flask server
cd backend
python3 main.py
```

### 3. **Frontend Setup (React)**

```bash
# Activate the virtual environment (if not already active)
source venv/bin/activate  # Windows: venv\Scripts\activate

# Navigate to the frontend directory
cd frontend

# Install frontend dependencies
npm install

# Start the development server
npm run dev
```

### 4. **Access the Application**

- Open your browser and navigate to `http://localhost:3000` to access the frontend.
- The backend API will run on `http://localhost:5173` by default.

### 5. **Common Issues & Fixes**

- **Issue:** Flask server not starting.  
  **Fix:** Ensure the virtual environment is activated and all dependencies are installed.

- **Issue:** React app not loading.  
  **Fix:** Ensure Node.js and npm are correctly installed. Run `npm install` again if needed.

---

## 📖 Usage Guide

### 1. **Select a Dataset**
- On the **Home Page**, choose from predefined datasets like **Iris**, **MNIST**, **Breast Cancer**, **California Housing**, or **CIFAR-10**.

### 2. **Build the Neural Network**
- Go to the **Build Page**.
- Drag and drop different layers (Dense, Convolutional, Dropout, etc.) from the sidebar onto the canvas.
- Connect layers logically to form a valid neural network structure.

### 3. **Configure Layer Parameters**
- Click on any layer to open the **Parameter Sidebar**.
- Customize settings like the number of neurons, activation functions, dropout rate, etc.

### 4. **Train the Model**
- Click the **Train** button.
- Monitor real-time progress through **interactive** graphs.
- View logs and metrics during the training process.

### 5. **Share the Trained Model**
- Once training is complete, go to the **Export Page**.
- Choose from multiple export formats:
  - **Python Script** (`.py`)
  - **Jupyter Notebook** (`.ipynb`)
  - **TensorFlow SavedModel**
  - **Keras Model** (`.keras`)
- Download the model for further use or deployment.

---

## 🔮 Future Enhancements

- **Custom Dataset Upload:** Enable users to upload their own datasets for model training.
- **PyTorch Integration:** Allow users to export models in PyTorch format.
- **Mobile Responsiveness:** Optimize the interface for mobile and tablet devices.
- **Advanced Visualizations:** Add more in-depth training visualizations (e.g., ROC Curve, Precision-Recall Curve).
- **Template Marketplace:** Introduce a marketplace for sharing and downloading user-created model templates.

---

## 🙏 Acknowledgments

- **React Flow:** For providing the core drag-and-drop functionality.
- **TensorFlow/Keras:** For model building and training.
- **Flask-SocketIO:** For real-time communication.
- Inspiration from existing neural network visual builders and educational tools.

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License
Permission is hereby granted, free of charge, to any person obtaining a copy of this software...
```
