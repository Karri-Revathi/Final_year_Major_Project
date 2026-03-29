from flask import Flask, request, render_template
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Importing hybrid model
from hybrid_model import HybridModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load Model
model = HybridModel()
model.load_state_dict(
    torch.load("Major_Project_Quantum_Inspired_CNN_8_epochs_weights.pth",
               map_location=torch.device('cpu'))
)

model.eval()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # change if your training used different size
    transforms.ToTensor(),
])

# Home Route
@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    prediction = None

    if request.method == 'POST':
        file = request.files['file']

        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Load image
            img = Image.open(filepath).convert('RGB')
            x = transform(img).unsqueeze(0)   # shape: [1, C, H, W]

            # Prediction
            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                pred_prob = probs[0][pred_class].item() * 100

                # Display Output
                if pred_class == 1:
                    result = "Pneumonia Detected"
                else:
                    result = "No Pneumonia Detected"

                prediction = {
                    "result": result,
                    "confidence": f"{pred_prob:.2f}"
                }

    return render_template("index.html", prediction=prediction)

# Run App
if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)