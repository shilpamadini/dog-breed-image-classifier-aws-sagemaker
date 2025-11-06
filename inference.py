import io, os, json
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

eval_transform = T.Compose([
    T.Resize(max(IMAGE_SIZE, 256)),
    T.CenterCrop(IMAGE_SIZE),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def net(num_classes: int, device: str):
    """
    '''
    Create a pretrained ResNet-50 model for transfer learning.
    Freezes all existing layers and replaces the final FC layer.
    Adjusts output size to match the number of classes.
    Moves the model to the specified compute device.
    """
    # weights=None at inference forces the model to rely only on your saved state_dict
    m = models.resnet50(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m.to(device)

# ---- SageMaker entry points ----
def model_fn(model_dir):
    """
    Reconstruct the model and load trained weights from model_dir/model.pth.
    Optionally loads labels.json if present to map class indices to names.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    state_path = os.path.join(model_dir, "model.pth")
    state = torch.load(state_path, map_location=device)

    # Infer num_classes from saved head
    if isinstance(state, dict) and "fc.weight" in state:
        num_classes = state["fc.weight"].shape[0]
    else:
        num_classes = 133  

    model = net(num_classes, device)
    model.load_state_dict(state)
    model.eval()

    labels_path = os.path.join(model_dir, "labels.json")
    idx_to_class = None
    if os.path.exists(labels_path):
        with open(labels_path, "r") as f:
            idx_to_class = json.load(f)

    model.idx_to_class = idx_to_class
    model.device = device
    return model

def input_fn(request_body, content_type="application/x-image"):
    """
    Accept raw image bytes (JPEG/PNG) and return a normalized tensor batch [1, 3, H, W].
    """
    if content_type in ("application/x-image", "image/jpeg", "image/png", "application/octet-stream"):
        img = Image.open(io.BytesIO(request_body)).convert("RGB")
        tensor = eval_transform(img).unsqueeze(0)
        return tensor
    raise ValueError(f"Unsupported content_type: {content_type}")

def predict_fn(input_data, model):
    """
    Run forward pass and return top-k predictions.
    """
    with torch.no_grad():
        input_data = input_data.to(model.device)
        logits = model(input_data)
        probs = torch.softmax(logits, dim=1)[0]
        topk = min(5, probs.numel())
        values, indices = torch.topk(probs, k=topk)
        values = values.tolist()
        indices = indices.tolist()
        if model.idx_to_class:
            labels = [model.idx_to_class.get(str(i), str(i)) for i in indices]
        else:
            labels = [str(i) for i in indices]
        return {"topk_indices": indices, "topk_labels": labels, "topk_probs": values}

def output_fn(prediction, accept="application/json"):
    if accept == "application/json":
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept: {accept}")
