from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from facenet_pytorch import MTCNN
import cv2
import tempfile
import os
import traceback

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, device=device)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

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
        return self.classifier(hidden[-1])

model = GRUClassifier()
model.load_state_dict(torch.load(r"D:\mini project\dfdc viva\DeepFake_Face_Detection\Backend\main_model_11.pth", map_location=device))
model.to(device)
model.eval()

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_faces_from_video(video_path, max_frames=30):
    cap = cv2.VideoCapture(video_path)
    faces = []
    count = 0
    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)
        if face is not None:
            faces.append(face)
            count += 1
    cap.release()
    return torch.stack(faces) if len(faces) == max_frames else None

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            path = tmp.name

        faces = extract_faces_from_video(path)
        if faces is None:
            os.remove(path)
            return JSONResponse(status_code=400, content={"error": "Insufficient face frames"})

        faces = torch.stack([transform(face) for face in faces]).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(faces)
            pred = torch.argmax(output, dim=1).item()
            label = "Fake" if pred == 1 else "Real"

        os.remove(path)
        return JSONResponse({"prediction": label})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
