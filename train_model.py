#Import dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import os, sys, logging, argparse
from PIL import ImageFile
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import smdebug.pytorch as smd

# Let PIL Handle partial images
ImageFile.LOAD_TRUNCATED_IMAGES = True

#Start Logging to capture DEBUG level messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Evaluation Function

def test(model, test_loader, criterion, hook=None):
    '''
    Evaluate the model on the test dataset.
    Runs model in evaluation mode on CPU.
    Computes average loss and accuracy without gradient updates.
    Prints performance metrics to console and logger.
    '''
    logger.info("Evaluation Start")
    test_loss = correct = 0

    if hook is not None:
        hook.set_mode(smd.modes.EVAL)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for (data, target) in test_loader:
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss = test_loss/len(train_loader.dataset)
    print(f"test_loss={test_loss:.6f};")
    print(f"test_accuracy={correct / len(test_loader.dataset)};")
    logger.info("Evaluation complete.")
    

def train(model, train_loader, valid_loader, criterion, optimizer, epochs, device, hook=None):
    '''
    Runs forward, backward, and optimization steps on the specified device.
    Train the model using training and validation data loaders.
    Tracks and prints average training and validation loss per epoch.
    '''
    logger.info("Training start.")
    
    if hook is not None:
        hook.set_mode(smd.modes.TRAIN)
                      
    for i in tqdm(range(epochs), desc="Training"):
        train_loss = 0
        model.train()

        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            if hook is not None:
                hook.record_tensor_value(tensor_name="train_loss", tensor_value=loss.item())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {i}: train_loss = {train_loss:.6f}")
        
        if hook is not None:
            hook.set_mode(smd.modes.EVAL)
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data = data.to(device)
                target = target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)
                val_loss += loss.item()
        print(f"val_loss={val_loss:.6f};")
        if hook is not None:
            hook.set_mode(smd.modes.TRAIN)
    logger.info("Training complete.")
        
    
def net(num_classes, device):
    '''
    Create a pretrained ResNet-50 model for transfer learning.
    Freezes all existing layers and replaces the final FC layer.
    Adjusts output size to match the number of classes.
    Moves the model to the specified compute device.
    '''
    logger.info("Model creation for fine-tuning.")
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(device)
    logger.info("Model creation complete.")
    return model

def create_data_loaders(data, batch_size, shuffle=True):
    '''
    Create a PyTorch DataLoader for batching and shuffling data.
    Organizes input data into mini-batches for efficient training.
    Randomizes sample order each epoch using shuffle=True.
    Returns a DataLoader ready for use in training loops.
    '''
    logger.info("Data loader creation start") 
    data_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    logger.info("Data loader creation complete") 
    return data_loader

def main(args):
    '''
    Initialize a model by calling the net function
    '''
    model = net(args.num_classes, args.device)
    
    '''
    Create loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    hook = None
    if smd is not None:
        try:
            hook = smd.Hook.create_from_json_file()
            hook.register_module(model)
            hook.register_loss(loss_criterion)
            logger.info("SageMaker Debugger hook registered.")
        except Exception as e:
            logger.info(f"Debugger hook not configured: {e}")
    else:
        logger.info("smd not available; proceeding without Debugger.")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    train_dir = os.path.join(args.data_dir, "train")
    valid_dir = os.path.join(args.data_dir, "valid")
    test_dir  = os.path.join(args.data_dir, "test")

    # Transform Input Image data ,ImageNet normalization for ResNet-50
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(max(args.image_size, 256)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_data = datasets.ImageFolder(valid_dir, transform=eval_transform)
    test_data  = datasets.ImageFolder(test_dir,  transform=eval_transform)

    train_loader = create_data_loader(train_data, args.batch_size, shuffle=True)
    valid_loader = create_data_loader(valid_data, args.batch_size, shuffle=False)
    test_loader  = create_data_loader(test_data, args.batch_size, shuffle=False)
    
    '''
    Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train(model, train_loader, valid_loader, loss_criterion, optimizer, args.epochs, args.device,hook=hook)
    
    '''
    Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion,hook=hook)
    
    '''
    Save the trained model
    '''
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(
        model.cpu().state_dict(),
        os.path.join(args.model_path, "model.pth")
    )
    logger.info("Model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train dog-breed classifier")

    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--num_classes", type=int, default=133, help="Number of target classes")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Training device")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size (short side after crop)")
    parser.add_argument("--model_path", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"),
                        help="Where to save model artifacts")
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING", "./data"),
                        help="Root dir containing train/ valid/ test/ subfolders")
    args, _ = parser.parse_known_args()
    main(args)
