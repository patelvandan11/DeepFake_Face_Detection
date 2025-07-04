from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from facenet_pytorch import MTCNN
import cv2
import tempfile
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import traceback

# -------------------- CONFIG --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, device=device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# -------------------- MODEL --------------------
class GRUClassifier(nn.Module):
    def __init__(self, feature_size=512, hidden_size=256, num_classes=2):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()
        self.gru = nn.GRU(input_size=feature_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.resnet(x)
        feats = feats.view(B, T, -1)
        _, hidden = self.gru(feats)
        out = self.classifier(hidden[-1])
        return out

# Load pretrained model
model = GRUClassifier()
model_path = r"D:\mini project\dfdc viva\DeepFake_Face_Detection\Backend\main_model_11.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# -------------------- UTILS --------------------
def extract_faces_from_video(video_path, mtcnn, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    faces = []
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(frame_rgb)
        if face is not None:
            faces.append(face)
            frame_count += 1
    cap.release()
    return torch.stack(faces) if len(faces) == max_frames else None

def extract_feature(video_path):
    faces = extract_faces_from_video(video_path, mtcnn)
    if faces is None:
        raise ValueError("Insufficient faces detected in the video.")
    faces = torch.stack([transform(face) for face in faces])
    faces = faces.unsqueeze(0).to(device)
    with torch.no_grad():
        B, T, C, H, W = faces.size()
        faces = faces.view(B * T, C, H, W)
        feats = model.resnet(faces)
        feats = feats.view(B, T, -1)
        avg_feat = feats.mean(dim=1).squeeze().cpu().numpy()
    return avg_feat

# -------------------- FASTAPI APP --------------------
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "DeepFake Detection API is running."}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename.endswith((".mp4", ".avi")):
            return JSONResponse(status_code=400, content={"error": "Only .mp4 or .avi files are supported."})

        # Save uploaded video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Extract faces
        faces = extract_faces_from_video(tmp_path, mtcnn)
        if faces is None:
            os.remove(tmp_path)
            return JSONResponse(status_code=400, content={"error": "Could not extract enough faces from video."})

        # Preprocess faces
        faces = torch.stack([transform(face) for face in faces])
        faces = faces.unsqueeze(0).to(device)  # [1, T, C, H, W]

        # Predict using GRUClassifier
        with torch.no_grad():
            output = model(faces)
            pred = torch.argmax(output, dim=1).item()
            label = "Fake" if pred == 1 else "Real"

        os.remove(tmp_path)

        return JSONResponse({"prediction": label})

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace": traceback.format_exc()}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)