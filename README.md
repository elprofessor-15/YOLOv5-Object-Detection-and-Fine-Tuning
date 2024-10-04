YOLOv5 Object Detection and Fine-Tuning

This repository showcases an implementation of object detection using the YOLOv5 model. It includes a complete workflow, starting from using a pre-trained YOLOv5 model to detect objects on custom images, to fine-tuning the model on custom datasets for improved accuracy and performance. The project also demonstrates how to visualize results and export detection outputs with bounding boxes and class labels.

Overview

The YOLO (You Only Look Once) family of models is well-known for real-time object detection with high accuracy. YOLOv5 builds on this legacy by offering a lightweight, fast, and highly efficient model that is easy to train and fine-tune on various datasets.

Key Features

	•	Pre-trained YOLOv5 Model: Utilizes the pre-trained YOLOv5 model to detect objects on a custom set of images.
	•	Fine-Tuning Capability: Fine-tunes the YOLOv5 model on custom datasets (e.g., KITTI dataset, COCO dataset) to enhance performance and adapt the model to specific object detection tasks.
	•	Detection Visualization: Visualizes the detected objects with bounding boxes and class labels on the images.
	•	Result Exporting: Provides a mechanism to export detection results, including annotated images and performance metrics.
Installation

To get started, clone the repository and install the necessary dependencies:

Clone the Repository: 

!git clone https://github.com/elprofessor-15/YOLOv5-Object-Detection-and-Fine-Tuning.git
cd YOLOv5-Object-Detection-and-Fine-Tuning

Install Dependencies

Use the following command to install the required Python libraries:

!pip install torch torchvision opencv-python matplotlib
!pip install -U 'git+https://github.com/ultralytics/yolov5.git'

Make sure you have Python 3.8+ and PyTorch installed.

Dataset Preparation

Using Pre-Trained YOLOv5 Model

The repository supports object detection using the pre-trained YOLOv5 model. You can easily run the model on your custom images.

	1.	Prepare your custom images and place them in the designated folder (e.g., data/images/).
	2.	Update the code to point to your input images.

Custom Dataset for Fine-Tuning

To fine-tune YOLOv5, you need a custom dataset in YOLO format. If you are working with datasets like COCO or KITTI, you can follow these steps:

	1.	Convert your dataset annotations to YOLO format.
	2.	Update the dataset configuration files (data.yaml) accordingly.

Usage

Running Object Detection with Pre-Trained YOLOv5

After setting up your environment, you can run object detection on custom images using the pre-trained YOLOv5 model:

from yolov5 import YOLOv5

# Initialize the YOLOv5 model
model = YOLOv5('yolov5s.pt', device='cuda')  # Use 'cuda' for GPU or 'cpu' for CPU

# Load your custom images
input_images = 'data/images/'

# Run detection and save results
results = model(input_images)
results.show()  # Visualize the detection results
results.save()  # Save the annotated images

Exporting Results

You can also export the detection results (bounding boxes, class labels, confidence scores, etc.):
results.print()  # Prints results to the console
results.save(save_dir='runs/detect/')  # Saves images with bounding boxes in 'runs/detect/'

Fine-Tuning on Custom Dataset

To fine-tune the YOLOv5 model on your custom dataset:

	1.	Prepare your dataset in YOLO format (images and labels).
	2.	Modify the data.yaml configuration file to specify the dataset path and class names.
	3.	Run the following command to fine-tune the model:
      python train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt
This command trains the model using 640x640 images, with a batch size of 16, for 50 epochs. You can adjust the parameters based on your dataset and resources.

Results and Visualization

Once the model completes object detection, you can view the results with bounding boxes and class labels drawn on each image. Additionally, you can inspect the performance metrics like precision, recall, and mean average precision (mAP).

You can visualize the results directly in the notebook or export them as images with the following commands:
results.show()  # Display images with detections
results.save()  # Save annotated images to the specified folder

Sample Outputs

Here are some example outputs showing how well the YOLOv5 model detects objects in images:

Object Detection on KITTI Images:

	•	Input Image: A sample image from the KITTI dataset.
	•	Detection Output: Bounding boxes with labels such as “car”, “pedestrian”, etc.

Fine-tuning the YOLOv5 model on the COCO dataset and then testing on 100 new images from the KITTI dataset improves detection accuracy, particularly for classes like cars and pedestrians in traffic scenes.

      
