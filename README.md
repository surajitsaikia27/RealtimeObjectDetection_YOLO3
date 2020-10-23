# RealtimeObjectDetection_YOLOv3: Creating a Realtime custom object Detector

Using this Repository you can build an object detector that can detect objects in real time based on webcam feed.
To train the detector we will be using Google open Image Dataset (https://opensource.google/projects/open-images-dataset).

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



## Start Training

To train your model, please type the following command from the root directory.

```
python /Training/TrainerOID.py

```
 
**To make everything run smoothly it is highly recommended to keep the original folder structure of this repo!**

The TrainerOID.py file has consist of various command-line options which you can modify in order to change epochs, learning rate, batch size etc. 
To speed up training, it is recommended to use a **GPU with CUDA** support. 

## Folder Structure   
   ```
   TrainYourOwnYOLO
    └─── Training
    └─── Deployment
    └─── Data
    └─── Utils
    ```
## Testing
Once you have trained the model, you will be needing a webcam to see how well the objects are detected.
The trained model will be saved as trained_weights_ODI.h5 in Model_weights folder.

To test the model, please type the following command from the root directory.


python /Deployment/videoRT.py

```

 
This repository  is modified from the following repository [github repo](https://github.com/AntonMu/TrainYourOwnYOLO)!
 

