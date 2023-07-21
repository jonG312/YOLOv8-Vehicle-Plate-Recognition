 # Automatic number-plate recognition (Model Training) 

&emsp;&emsp;Automatic Number Plate Recognition (ANPR), also known as License Plate Recognition (LPR), is a technology that uses optical character recognition (OCR) and computer vision to automatically read and interpret vehicle registration plates. The system captures images of vehicles' number plates using cameras, processes the images to extract the alphanumeric characters on the plate, and then converts them into computer-readable text.

&emsp;&emsp;ANPR systems are widely used for various purposes, such as identifying vehicles involved in criminal activities, tracking stolen vehicles, enforcing traffic rules (e.g., speed limits and red-light violations), managing parking facilities, and collecting tolls on highways. However, the technology raises concerns about privacy and data security, as it involves the collection and processing of potentially sensitive information. Therefore, its deployment and usage are subject to regulations and safeguards in many jurisdictions.

## Implementation of the YOLOv8 model

&emsp;&emsp;Implementing YOLO for Automatic Number Plate Recognition (ANPR) involves training a YOLO model on a custom dataset of license plate images and then integrating it with an OCR (Optical Character Recognition) system to read the characters from the detected license plate regions 

**steps involved:**

- `Dataset Collection:` Collect a dataset of annotated license plate images. The dataset should contain images of vehicles with annotated bounding boxes around the license plates and corresponding alphanumeric characters.

![DataCollection](https://github.com/jonG312/YOLOv8-Vehicle-Plate-Recognition/blob/main/YOLOv8_custom_data_set/data/runs/detect/train4/train_batch1520.jpg)


- `Data Preprocessing:` Preprocess the dataset by resizing the images to a consistent resolution (e.g., 416x416 pixels), normalizing pixel values, and preparing the annotations in the YOLO format (x, y, width, height) normalized relative to the image size.

![annotations](https://github.com/jonG312/YOLOv8-Vehicle-Plate-Recognition/blob/main/YOLOv8_custom_data_set/Resources/annotations.png)

- `Model Selection:` This model is trained with the YOLOv8 algorithm.

- `Model Architecture:` Set up the YOLO architecture with the appropriate number of output layers to predict bounding boxes and class probabilities. The last layer's number of neurons should match the total number of classes you are detecting (in this case, the number of alphanumeric characters).

**google_colab_config.yaml:**

```
path: '/content/drive/My Drive/YOLOv8_custom_data_set/data/' # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/train  # val images (relative to 'path')

# Classes
names:
  0: vehicle registration plate
  1: vehicle
```

- `Train the YOLO Model:` Train the YOLO model on the custom dataset using a deep learning framework like TensorFlow or PyTorch. Fine-tune the pre-trained model on your ANPR dataset to achieve better performance.


**YOLOv8 Custom Data-set**

<p align="left">
<a href="https://colab.research.google.com/github/jonG312/YOLOv8-Vehicle-Plate-Recognition/blob/main/YOLOv8CustomDataSet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
</p>

**Setting Environment:**

```
# importing GPU

import tensorflow as tf
tf.test.gpu_device_name()    

# verifying GPU
!/opt/bin/nvidia-smi
```

**Output:**    
```
Thu Jul 20 22:02:33 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.105.17   Driver Version: 525.105.17   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
| N/A   42C    P0    25W /  70W |    387MiB / 15360MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
+-----------------------------------------------------------------------------+
```

```
from tensorflow.python.client import device_lib
device_lib.list_local_devices()
```
```     
# Mounting google drive

from google.colab import drive
drive.mount('/content/drive')
```
```    
ROOT_DIR = '/content/drive/My Drive/YOLOv8_custom_data_set/'
     
%cd /content/drive/My Drive/YOLOv8_custom_data_set/
! ls
```
**Installing Dependencies:**
```
# Pip install ultralytics and dependencies and check software and hardware.
%pip install ultralytics
import os
from ultralytics import YOLO
```
**Trainning Model:**
```
from ultralytics import YOLO
# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from scratch
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data=os.path.join(ROOT_DIR, "google_colab_config.yaml"), epochs=200)  # train the model
```
```
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding
```
**Pushing run into the folder data:**    

```
!scp -r /content/runs '/content/gdrive/My Drive/YOLOv8_custom_data_set/data/
```


- `Post-processing:` After obtaining the bounding box predictions from the YOLO model, perform non-maximum suppression (NMS) to filter out overlapping and low-confidence detections.

- `License Plate Region Cropping:` For each remaining bounding box after NMS, crop the corresponding region from the original image. This region will contain the license plate.

- `OCR Integration:` Pass each cropped license plate region through an OCR system (e.g., Tesseract or any other OCR library) to read the alphanumeric characters from the license plate.

- `Interpretation and Usage:` The OCR output will provide the recognized characters from the license plate. You can use this information for various applications like vehicle tracking, parking management, toll collection, etc.




