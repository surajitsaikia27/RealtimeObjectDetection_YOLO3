# RealtimeObjectDetection_YOLOv3: Creating a Realtime custom object Detector

Using this Repository you can build an object detector that can detect objects in real time based on webcam feed.
To train the detector we will be using Google open Image Dataset (https://opensource.google/projects/open-images-dataset)
This repository has been tested using TensorFlow > 2 and Cuda 11.

### Download the Dataset for training

At First we need to download the dataset for training. 
Please refer to this link to know how to download the dataset and annotate it.


## Repo structure
+ [`Training`](/Training/): Scripts to train the YOLOv3 model
+ [`Deployment`](/Deployment/): Script to run the trained YOLO detector using webcam
+ [`Data`](/Data/): Input Data, Output Data, Model Weights and Results
+ [`Utils`](/Utils/): Utility scripts used by main scripts


### Installation


Clone this repo with:
```
Download or clone this repository
cd RealtimeObjectDetection_YOLOv3/
```

#### Install dependencies [Windows, Mac or Linux]
```
pip install -r requirements.txt
```

## Copy the dataset
Copy the Training images and the trainannotation.txt file to the following path:
./Data/Source_Images/OIDsample/train


## Start Training

To train your model, please type the following command from the root directory.

```
python /Training/OIDtrain.py

```
 
**To make everything run smoothly it is highly recommended to keep the original folder structure of this repo!**

Each `OIDtrain.py` file has various command line options that help tweak performance and change things such as input and output directories. All scripts are initialized with good default values that help accomplish all tasks as long as the original folder structure is preserved.

To speed up training, it is recommended to use a **GPU with CUDA** support. 

## Troubleshooting
If you encounter a `FileNotFoundError`, `Module not found` or similar error, make sure that you did not change the folder structure. Your directory structure **must** look exactly like this: 
   
   ```
    TrainYourOwnYOLO
    └─── Training
    └─── Deployment
    └─── Data
    └─── Utils
    ```
    Note: Please specify the correct paths related to the training images

## Testing
Once you have trained the model, you will be needing a webcam to see how well the objects are detected.

To test the model, please type the following command from the root directory.

```
python /Deployment/videoRT.py

```

 
This repository  is modified from the following repository [github repo](https://github.com/AntonMu/TrainYourOwnYOLO)!
 

