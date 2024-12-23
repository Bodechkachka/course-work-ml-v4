import os
from flask import Flask, request, jsonify, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from PIL import Image
from pathlib import Path

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = Path("static/results")
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Загрузка MobileNet
transfer_model = models.mobilenet_v2(pretrained=False)
# Загруза YOLOv5
model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/exp2/weights/best.pt', source='local')

transfer_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
for param in transfer_model.parameters():
    param.requires_grad = False

num_classes = 5
transfer_model.classifier[1] = nn.Linear(transfer_model.classifier[1].in_features, num_classes)

def preprocess_image(image_path):
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize((640, 640))
    return image


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(819200, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

cnn_model = SimpleCNN(num_classes=5)
cnn_model.load_state_dict(torch.load("D:/Machine-learning/course-work-ml/cnnmodel.pth"))
cnn_model.eval()

transfer_model.load_state_dict(torch.load("D:/Machine-learning/course-work-ml/mobilenet_transfer_learning.pth"))
transfer_model.eval()

cnn_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

classes = ['axe', 'bow', 'hammer', 'mace', 'sword']


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cnn')
def cnn():
    return render_template('CNN.html')

@app.route('/transfer-learning')
def transferLearning():
    return render_template('transfer-learning.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        preprocessed_image = preprocess_image(file_path)
        preprocessed_image.save(file_path)

        preprocessed_image_path = os.path.splitext(file_path)[0] + ".jpg"
        preprocessed_image.save(preprocessed_image_path)

        results = model(file_path)

        result_image_path = RESULT_FOLDER / os.path.basename(os.path.splitext(filename)[0] + ".jpg")
        results.save(save_dir=RESULT_FOLDER, exist_ok=True)

        result_image_path_str = f"/static/results/{os.path.basename(result_image_path)}"
        return render_template('index.html', result_image=result_image_path_str)

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = Image.open(file_path).convert('RGB')
        image = cnn_transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = cnn_model(image)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = classes[predicted.item()]

        return render_template('CNN.html', classification_result=predicted_class)

@app.route('/transfer-classify', methods=['POST'])
def transfer_classify_image():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        image = Image.open(file_path).convert('L')
        image = cnn_transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = cnn_model(image)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = classes[predicted.item()]

        return render_template('transfer-learning.html', classification_result=predicted_class)

@app.route('/api', methods=['POST'])
def api():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        preprocessed_image = preprocess_image(file_path)

        preprocessed_image_path = os.path.splitext(file_path)[0] + ".jpg"
        preprocessed_image.save(preprocessed_image_path)

        results = model(preprocessed_image_path)

        detected_classes = results.pandas().xyxy[0]['name'].tolist()  # Список классов

        return jsonify({'detected_classes': detected_classes})

if __name__ == '__main__':
    app.run(debug=True)