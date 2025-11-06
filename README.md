# Dog Breed Classification with AWS SageMaker

##  Introduction
This project trains, tunes, and deploys a ResNet-50 convolutional neural network to classify dog breeds.  
It demonstrates a complete end-to-end ML workflow on Amazon SageMaker

The end-to-end ML workflow is built entirely on Amazon SageMaker, leveraging:

Hyperparameter Optimization (HPO) for fine-tuning model performance.

SageMaker Debugger & Profiler for detecting system bottlenecks and training inefficiencies.

Model Deployment to a real-time inference endpoint for prediction.

The model can identify one of 133 dog breeds from a single image and return top-k probabilities.


# Project Setup Instructions

## Environment Setup

```
pip install sagemaker smdebug boto3 torch torchvision
```

## Data Download

```
!wget https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
!unzip dogImages.zip
```

## The dataset contains folders for train, valid, and test.

### Upload to S3

```
from sagemaker import Session
sess = Session()
S3_BUCKET = sess.default_bucket()
DATA_PREFIX = "dogimages"
input_data_path = sess.upload_data(path="dogImages", bucket=S3_BUCKET, key_prefix=DATA_PREFIX)
```
![S3 Folders](https://github.com/shilpamadini/dog-breed-image-classifier-aws-sagemaker/blob/53cbf6088f654b8fc7d8142d0ddfc5d5c09242ab/images/s3%20bucket%20folder%20structure%20.png)

### Training Script Files

train_model.py → Main training script for profiling and debugging.

hpo.py → Training script used for Hyperparameter Tuning.

inference.py → Defines model loading, input/output handlers, and prediction logic.

train_deploy.ipynb → Jupyter notebook orchestrating data prep, training, tuning, profiling, and deployment.

model.pth → Saved trained weights.

labels.json → Class index-to-name mapping for readable predictions.

##  Model Training and Hyperparameter Tuning

## Key Tuned Hyperparameters

| Hyperparameter | Range Tested | Best Value |
|--------|--------------|--------|
| Learning Rate | [1e-4, 1e-2] | 0.0044 |
|Batch Size | [8, 32] | 29 |
|Epochs | [3, 10] | 9 |

Objective metric: val_loss

![training jobs](https://github.com/shilpamadini/dog-breed-image-classifier-aws-sagemaker/blob/ba0a09ce5227cea8a9814e81b78736704a197442/images/training%20jobs.png))


The best model was automatically selected by SageMaker.
![Hyperparameters of Best Training Job](https://github.com/shilpamadini/dog-breed-image-classifier-aws-sagemaker/blob/ba0a09ce5227cea8a9814e81b78736704a197442/images/Hyperparameters%20of%20best%20training%20job.png))

## Training Summary

| Metric |Value |
|--------|--------------|
| Epochs | 8 completed |
|Final Validation Loss | 13.28 | 
|Final Test Loss | 0.01569 | 
|Final Test Accuracy | ~86.8% | 
|Training Time | ~24 mins (1437 sec) | 

This shows the model trained smoothly, converged, and validated well.

### Model Behavior Observations
1. Training loss steadily dropped epoch-to-epoch.

2. Validation loss also declined consistently.
3. ~87% on the dog breed task (133 classes!) 

## Profiler Highlights:

GPUMemoryIncrease triggered 95 times (mild fluctuation, normal behavior).

LowGPUUtilization / BatchSize triggered 15–16 times, GPU underutilized due to small batch size or CPU delays.

CPUBottleneck triggered 5 times → data preprocessing slower than GPU compute.

Dataloader rule flagged only 1 worker for 4 CPU cores.

No issues found for Overfit, VanishingGradient, or LoadBalancing.

Expected to find issues in PoorWeightInitialization since pretrained model with frozen layers triggers heuristic

![Timeline Charts]([image-url "Optional title"](https://github.com/shilpamadini/dog-breed-image-classifier-aws-sagemaker/blob/ba0a09ce5227cea8a9814e81b78736704a197442/images/Profiler%20Timeline%20Charts.png))



## Model Deployment

After training, the model artifact (model.tar.gz) was deployed to a real-time endpoint using:

```
from sagemaker.pytorch import PyTorchModel

model = PyTorchModel(
    entry_point="inference.py",
    source_dir=".",
    role=role,
    model_data=s3_model_path,
    framework_version="1.13",
    py_version="py39"
)

predictor = model.deploy(initial_instance_count=1, instance_type="ml.m5.xlarge")
```
![Model End Point](https://github.com/shilpamadini/dog-breed-image-classifier-aws-sagemaker/blob/ba0a09ce5227cea8a9814e81b78736704a197442/images/model%20end%20point.png))


## Inference Example

An image from S3 was sent to the endpoint:
s3://sagemaker-us-east-1-106660882488/dogimages/test/001.Affenpinscher/Affenpinscher_00003.jpg

```
import boto3
import io
from PIL import Image

s3 = boto3.client("s3")
bucket = "sagemaker-us-east-1-106660882488"
key = "dogimages/test/001.Affenpinscher/Affenpinscher_00003.jpg"

img_bytes = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
result = predictor.predict(img_bytes)
print(result)

```

## Output

```
{
  "topk_labels": [
    "001.Affenpinscher",
    "042.Cairn_terrier",
    "036.Briard",
    "026.Black_russian_terrier",
    "033.Bouvier_des_flandres"
  ],
  "topk_probs": [
    0.9820, 0.0055, 0.0027, 0.0023, 0.0019
  ]
}

```
Prediction: The model correctly classified the image as an **Affenpins
