import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import sys
import io
from io import BytesIO

# Fix encoding issue for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)

# Lazy load model (to avoid startup delay)
model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model

    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    try:
        # Load the model only once (lazy load)
        if model is None:
            model = load_model('model/my_model.keras')

        # Convert the uploaded file to BytesIO
        img_bytes = BytesIO(file.read())
        
        # Load and preprocess the image for prediction
        img = image.load_img(img_bytes, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make the prediction using the loaded model
        prediction = model.predict(img_array)

        # Determine the result
        result = "It's a Dog!" if prediction[0] > 0.5 else "It's a Cat!"
        return render_template('result.html', result=result)
    
    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == "__main__":
    # Use PORT environment variable for Azure compatibility
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
