from flask import Flask,request,jsonify
from flask_cors import CORS
#for model
import torch # type: ignore
from torchvision import transforms # type: ignore
import torchvision.transforms # type: ignore
from PIL import Image
from io import BytesIO
import base64
from typing import List,Tuple
import os
import gdown # type: ignore



app=Flask(__name__)
CORS(app)

#device(GPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class_names = ['Apple_Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Blueberry_healthy', 'Cherry(including_sour)Powdery_mildew', 'Cherry(including_sour)healthy', 'Corn(maize)Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust', 'Corn(maize)Northern_Leaf_Blight', 'Corn(maize)healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Grapehealthy', 'Orange_Haunglongbing(Citrus_greening)', 'PeachBacterial_spot', 'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato_healthy']

# === Download model from Google Drive if not present ===
model_path = "full_model.pth"
model_drive_url = "https://drive.google.com/uc?id=1DXpL1anOs6943Ifj1Uno7_4nd99RjGU3"

if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(model_drive_url, model_path, quiet=False)

# Load model
model = torch.load(model_path, weights_only=False)

#transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


#prediction
def pred(model: torch.nn.Module,image_path: bytes,class_names: List[str],image_size: Tuple[int, int] = (299, 299),transform: torchvision.transforms = None,device: torch.device=device):
    
    img = Image.open(BytesIO(image_path))

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    model.to(device)

    model.eval()
    with torch.inference_mode():
      transformed_image = image_transform(img).unsqueeze(dim=0)

      target_image_pred = model(transformed_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
    output =class_names[target_image_pred_label]
    return output


@app.route('/')
def home():
    return "welcome to the crop prediction deep learning API"


@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content for favicon

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    base64img = data.get('image')

    if not base64img:
        return jsonify({'error': 'No image data found'}), 400

    try:
        image_data = base64.b64decode(base64img)
        op = pred(
            model=model,
            image_path=image_data,
            class_names=class_names,
            transform=transform,
            image_size=(224, 224)
        )
        return jsonify({'status':'ok','predicted_class': op}), 200

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Failed to process image'}), 500

if __name__ == '__main__':
    port=int(os.environ.get("PORT",10000))
    app.run(host='0.0.0.0',port=port)