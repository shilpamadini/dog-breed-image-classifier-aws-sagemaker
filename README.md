# Dog Breed Classification with AWS SageMaker

## Overview

Fine tuning a pre-trained ResNet model for dog breed classification using AWS SageMaker. The project demonstrates a full MLOps workflow from data preparation to deployment, including hyperparameter tuning, profiling, and real-time inference.The model is trained using transfer learning to classify dog breeds from images, monitored with SageMaker Debugger and Profiler, and deployed to a live endpoint for predictions.

## Steps:

1. Prepared and Uploaded Data to S3
Downloaded the dog breed classification dataset (available online as a public example dataset for image classification), organized it into training and validation splits, and uploaded it to an S3 bucket for SageMaker access.

2. Fine Tuned a Pre Trained ResNet Model
Used transfer learning to improve model performance on dog breed classification.

3. Configured and Launched SageMaker Training Jobs
Tracked metrics, saved artifacts, and ensured reproducibility using training scripts.

4. Ran Hyperparameter Tuning (HPO)
Used SageMaker Tuner to search for optimal learning parameters and selected the best performing model configuration.

5. Enabled Debugger and Profiler
Monitored training behavior, performance bottlenecks, and system utilization for optimization insights.

6. Deployed Trained Model to a SageMaker Endpoint
Packaged the best model and created a managed inference endpoint.

7. Performed Real Time Inference on Test Images

8. Queried the endpoint from the inference notebook and validated predictions.
