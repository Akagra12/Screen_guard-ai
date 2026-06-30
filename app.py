import os
import tempfile
from flask import Flask, request, jsonify
from predict import predict

app = Flask(__name__)

# Premium, self-contained HTML/CSS/JS frontend served directly from the backend
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spot the Fake Photo - Recapture Detector</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: rgba(30, 41, 59, 0.7);
            --border-color: rgba(255, 255, 255, 0.1);
            --primary-accent: #8b5cf6;
            --success-color: #10b981;
            --danger-color: #ef4444;
            --text-color: #f8fafc;
            --text-secondary: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Outfit', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 0% 0%, rgba(139, 92, 246, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%);
            background-attachment: fixed;
            color: var(--text-color);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem 1rem;
        }

        .container {
            max-width: 650px;
            width: 100%;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
            text-align: center;
        }

        h1 {
            font-weight: 700;
            font-size: 2.2rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #fff 30%, var(--primary-accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            color: var(--text-secondary);
            margin-bottom: 2rem;
            font-size: 1.05rem;
            font-weight: 300;
        }

        /* Upload Zone */
        .upload-zone {
            border: 2px dashed rgba(139, 92, 246, 0.3);
            border-radius: 16px;
            padding: 3rem 2rem;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            background: rgba(15, 23, 42, 0.4);
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }

        .upload-zone:hover {
            border-color: var(--primary-accent);
            background: rgba(139, 92, 246, 0.05);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(139, 92, 246, 0.15);
        }

        .upload-icon {
            font-size: 3rem;
            color: var(--primary-accent);
            animation: bounce 2s infinite;
        }

        .upload-text {
            color: var(--text-color);
            font-weight: 600;
            font-size: 1.1rem;
        }

        .upload-subtext {
            color: var(--text-secondary);
            font-size: 0.875rem;
        }

        #file-input {
            display: none;
        }

        /* Image Preview */
        .preview-container {
            display: none;
            width: 100%;
            margin-top: 1.5rem;
        }

        .preview-img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            object-fit: contain;
        }

        /* Result Dashboard */
        .result-container {
            display: none;
            margin-top: 2rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border-color);
            animation: fadeIn 0.5s ease-out;
        }

        /* Score Gauge */
        .gauge-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 1.5rem 0;
        }

        .gauge-label {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            letter-spacing: 0.5px;
        }

        .gauge-score {
            font-size: 3.5rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 0.25rem;
        }

        .verdict {
            display: inline-block;
            padding: 0.5rem 1.5rem;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            margin-top: 0.5rem;
        }

        .verdict.real {
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--success-color);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }

        .verdict.fake {
            background-color: rgba(239, 68, 68, 0.15);
            color: var(--danger-color);
            border: 1px solid rgba(239, 68, 68, 0.3);
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1.5rem;
        }

        .metric-card {
            background: rgba(15, 23, 42, 0.5);
            border: 1px solid var(--border-color);
            padding: 1rem;
            border-radius: 12px;
            text-align: center;
        }

        .metric-value {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-color);
        }

        .metric-title {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-top: 0.25rem;
        }

        /* Buttons */
        .reset-btn {
            background: transparent;
            border: 1px solid var(--border-color);
            color: var(--text-color);
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            cursor: pointer;
            margin-top: 1.5rem;
            font-weight: 600;
            transition: all 0.2s;
        }

        .reset-btn:hover {
            background: rgba(255,255,255,0.05);
            border-color: var(--text-secondary);
        }

        /* Loader */
        .loader {
            display: none;
            margin: 2rem 0;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }

        .spinner {
            width: 48px;
            height: 48px;
            border: 3px solid rgba(139, 92, 246, 0.2);
            border-top-color: var(--primary-accent);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-6px); }
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .error-message {
            display: none;
            color: var(--danger-color);
            margin-top: 1rem;
            font-weight: 500;
        }
    </style>
</head>
<body>

<div class="container">
    <h1>Spot the Fake Photo</h1>
    <p class="subtitle">Upload or drop any image to detect if it's a real photo or a screen recapture.</p>
    
    <!-- Upload Form -->
    <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
        <span class="upload-icon">⚡</span>
        <span class="upload-text">Drag & Drop Image Here</span>
        <span class="upload-subtext">or click to browse from device</span>
        <input type="file" id="file-input" accept="image/*" onchange="handleFileSelect(event)">
    </div>

    <!-- Error Message -->
    <div class="error-message" id="error-box"></div>

    <!-- Loader -->
    <div class="loader" id="loader">
        <div class="spinner"></div>
        <p style="color: var(--text-secondary)">Running fraud pipeline algorithms...</p>
    </div>

    <!-- Preview Container -->
    <div class="preview-container" id="preview-box">
        <img class="preview-img" id="preview-img" src="" alt="Selected Preview">
    </div>

    <!-- Results Panel -->
    <div class="result-container" id="result-box">
        <div class="gauge-container">
            <span class="gauge-label">Fraud Score</span>
            <span class="gauge-score" id="score-text">0.00%</span>
            <span class="verdict" id="verdict-badge">REAL PHOTO</span>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <span class="metric-value">~330 ms</span>
                <span class="metric-title">Inference Time</span>
            </div>
            <div class="metric-card">
                <span class="metric-value">$0.00</span>
                <span class="metric-title">On-Device Cost</span>
            </div>
        </div>

        <button class="reset-btn" onclick="resetDemo()">Analyze Another Photo</button>
    </div>
</div>

<script>
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const previewBox = document.getElementById('preview-box');
    const previewImg = document.getElementById('preview-img');
    const resultBox = document.getElementById('result-box');
    const loader = document.getElementById('loader');
    const errorBox = document.getElementById('error-box');
    const scoreText = document.getElementById('score-text');
    const verdictBadge = document.getElementById('verdict-badge');

    // Drag-and-drop events
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary-accent)';
        dropZone.style.background = 'rgba(139, 92, 246, 0.08)';
    }

    function unhighlight(e) {
        e.preventDefault();
        dropZone.style.borderColor = 'rgba(139, 92, 246, 0.3)';
        dropZone.style.background = 'rgba(15, 23, 42, 0.4)';
    }

    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            fileInput.files = files;
            processImage(files[0]);
        }
    }

    function handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            processImage(file);
        }
    }

    function processImage(file) {
        errorBox.style.display = 'none';
        
        // Show Image Preview
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function() {
            previewImg.src = reader.result;
            previewBox.style.display = 'block';
            dropZone.style.display = 'none';
            
            // Run prediction request
            uploadAndPredict(file);
        }
    }

    function uploadAndPredict(file) {
        loader.style.display = 'flex';
        resultBox.style.display = 'none';

        const formData = new FormData();
        formData.append('image', file);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loader.style.display = 'none';
            if (data.error) {
                showError(data.error);
                return;
            }
            displayResult(data.score);
        })
        .catch(err => {
            loader.style.display = 'none';
            showError("Network connection error. Is the server running?");
        });
    }

    function displayResult(score) {
        const percentage = (score * 100).toFixed(2);
        scoreText.innerText = percentage + "%";
        
        // SVM decision boundary is at 0.5
        if (score >= 0.5) {
            verdictBadge.innerText = "SCREEN RECAPTURE";
            verdictBadge.className = "verdict fake";
            scoreText.style.color = "var(--danger-color)";
        } else {
            verdictBadge.innerText = "REAL PHOTO";
            verdictBadge.className = "verdict real";
            scoreText.style.color = "var(--success-color)";
        }
        
        resultBox.style.display = 'block';
    }

    function showError(message) {
        errorBox.innerText = message;
        errorBox.style.display = 'block';
        resetDemo();
    }

    function resetDemo() {
        dropZone.style.display = 'flex';
        previewBox.style.display = 'none';
        resultBox.style.display = 'none';
        loader.style.display = 'none';
        fileInput.value = '';
    }
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return INDEX_HTML


@app.route("/predict", methods=["POST"])
def run_prediction():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
        
    try:
        # Create a temporary file to save the uploaded image
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, file.filename)
        file.save(temp_path)
        
        # Run model prediction
        score = predict(temp_path)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return jsonify({"score": score})
    except Exception as e:
        return jsonify({"error": f"Error running detection: {str(e)}"}), 500


if __name__ == "__main__":
    print("\\n=======================================================")
    print("  SPOT THE FAKE PHOTO - WEB DEMO SERVER                 ")
    print("=======================================================")
    print("Starting local server...")
    print("Open http://127.0.0.1:5000 in your web browser!\\n")
    
    # Run locally
    app.run(host="127.0.0.1", port=5000, debug=False)
