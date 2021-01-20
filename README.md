# RoboVec:2.0

In this project, a home robot known as Vector is programmed to do few tasks based on voiced commands. It is uses deep learning, computer vision and speech recognision, and is programmed using the Vector SDK. However, in case if you dont have a Vector, then using this repository you can create your own custom object detector which can be tested using webcam.

The Vector SDK gives access to various capabilities of this robot, such as computer vision, Artificial intelligence, navigation and etc. You can design your own programs to make this robot pet imbibed with AI capabilities. Here, in this project, I have trained and used an real time object detector which lets the robot to recognise objects in its surrounding environment. Moreover, in this module, instruction are provided how to create your own customize object detector. In case, if you want train our own object detector then follow the instructions below or else it can be skipped since the pre-trained model can be downloaded.

### Train the object detector in custom dataset
Please refer to the following link to know how to download the dataset and annotate it.
https://surjeetsaikia.medium.com/build-your-realtime-custom-object-detector-using-yolov3-f61af825153f

## Start Training

To train your model, please type the following command from the root directory after downloading the dataset.

```
python ./Training/TrainerOID.py

```
The TrainerOID.py file has consist of various command-line options which you can modify in order to change epochs, learning rate, batch size etc. 
To speed up training, it is recommended to use a **GPU with CUDA** support. 



Before running this module, install the vector SDK by following the information in this page:
Using this Repository you can build an object detector that can detect objects in real time based on webcam feed.
To train the detector we will be using Google open Image Dataset (https://opensource.google/projects/open-images-dataset).

This repository has been tested using TensorFlow > 2 and Cuda 11.
## Repo structure
+ [`Training`](/Training/): Scripts to train the Object detector. However, it can be skipped if you want to test with a trained detector
+ [`Deployment`](/Deployment/): Script to test the robot.
+ [`Data`](/Data/): Input Data, Output Data, Model Weights and Results
+ [`Utils`](/Utils/): Utility scripts used by main scripts
+ [`vector-python-sdk-master`](vector-python-sdk-master): This repository contains the SDK files of the vector robot. However, it is not provided due to its larger size. Please follow the instructions below in order to install the SDK. 

**To make everything run smoothly it is highly recommended to keep the original folder structure of this repo!**
### Installation


Clone or download this repo with:
```
cd RoboVec/
```

#### Install dependencies [Windows, Mac or Linux]
```
pip install -r requirements.txt
```

#### Installing Vector SDK
Install the vector SDK by following the information in this page:
(https://developer.anki.com/vector/docs/index.html).



## Testing (if you have trained the object detector)
Once you have trained the model, you will be needing a webcam to see how well the objects are detected.
The trained model will be saved as trained_weights_ODI.h5 in Model_weights folder.

To test the model, please type the following command from the root directory.


```
python ./Deployment/videoRT.py

```
## Test Vector
Before running the module, you need to authenticate the vector robot. To authenticate with the robot, type the following into the Terminal window.
```
python3 -m anki_vector.configure
```
Please note that the robot and your computer should be connected to the same network. Now, you will be asked to enter your robot’s name, ip address and serial number, which you can find in the robot itself. Also, You will be asked for your Anki login and password which you used to set up your Vector.
If the installation is successfull, then Vector can be brought to life and he can obey your voice commands.

```
python ./Deployment/MultifunctionVec.py

```


### The Vector Class
The class contains the main functionalities that drives vector into action. However, you can change the voice commands as you want and new functionalities can be added.

```python
class VecRobot:

    def __init__(self):
        self.name = 'vector'
        self.data_folder = os.path.join(get_parent_dir(n=1), "Data")
        self.model_folder = os.path.join(self.data_folder, "Model_Weights")
        self.model_weights = os.path.join(self.model_folder, "yolo.h5")
        self.model_classes = os.path.join(self.model_folder, "data_coco.txt")
        self.anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

    def get_classnames(self, classes):
        try:
            class_list = []
            for class_names in classes:
               class_list.append(class_names)
            class_list=set(class_list)
            return ', '.join(class_list)
        except:
            return 'no objects'

    def vector_speaks(self, text):
        robot = anki_vector.Robot(anki_vector.util.parse_command_args().serial)
        robot.connect()
        print('Vector: {}'.format(text))
        robot.behavior.say_text(text)
        robot.disconnect()

    def speech_regognizer(self, recognizer, microphone):
        """Transcribe speech from recorded from `microphone`.
        """

        if not isinstance(recognizer, sr.Recognizer):
            raise TypeError("`recognizer` must be `Recognizer` instance")

        if not isinstance(microphone, sr.Microphone):
            raise TypeError("`microphone` must be `Microphone` instance")

        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)

        response = {
            "success": True,
            "error": None,
            "listened": None
        }
        try:
            response["listened"] = recognizer.recognize_google(audio)
        except sr.RequestError:
            response["success"] = False
            response["error"] = "API unavailable"
        except sr.UnknownValueError:
            response["error"] = "Unable to recognize speech"
        return response

    def obstacle_detection(self):
        with anki_vector.Robot() as robot:
            robot.motors.set_head_motor(-3.0)
            i = 1
            deq = collections.deque([15, 15, 15], 3)
            while True:
                distance = robot.proximity.last_sensor_reading.distance
                deq.append(distance.distance_inches)
                print('Distance reading DEQUE', sum(deq) / 3)
                robot.motors.set_wheel_motors(50, 50)

                if sum(deq) / 3 <= 3.5:
                    robot.motors.stop_all_motors()
                    robot.behavior.turn_in_place(degrees(-60))
                else:
                    robot.behavior.drive_straight(distance_mm(100), speed_mmps(80))
                    robot.behavior.turn_in_place(degrees(60))

    def real_time_detection(self, param=None):
        anchors = get_anchors(anchors_path)

        # labels to draw on images
        class_file = open(model_classes, "r")
        input_labels = [line.rstrip("\n") for line in class_file.readlines()]
        # print("Found {} input labels: {} ...".format(len(input_labels), input_labels))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        thickness = 1
        class_detected = []
        k=0
        with anki_vector.Robot() as robot:

            if param == 'car':
                robot.motors.set_head_motor(-1.0)
            while k < 60:
              k=k+1
              if k%10==0:
                robot.behavior.turn_in_place(degrees(60))
              else:
                print(k)
                robot.camera.init_camera_feed()
                img = cv2.cvtColor(np.array(robot.camera.latest_image.raw_image), cv2.COLOR_RGB2BGR)
                img2 = Image.fromarray(img)
                bbox, class_name, score = detect_frame(
                    yolo,
                    img2
                )

                if param in class_name:
                    return param
                c = cv2.waitKey(1)
                if c == 27:
                    break
                if len(bbox) != 0:
                    color = list(np.random.random(size=3) * 256)
                    for i in range(len(bbox)):
                        cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color, 2)
                        class_detected.append(input_labels[class_name[i]])
                        if param in class_detected:
                            cv2.destroyAllWindows()
                            return param
                        cv2.putText(img, input_labels[class_name[i]] + '  Prob:' + str(score[i]),
                                    (bbox[i][0], bbox[i][1]), font,
                                    fontScale, color, thickness, cv2.LINE_AA)

                cv2.imshow('output', img)
            cv2.destroyAllWindows()
        end = timer()
        return class_detected

    def display_image(self, file_name):
        with anki_vector.Robot() as robot:
            print('display image = {}'.format(file_name))
            image = Image.open(file_name)
            screen_data = anki_vector.screen.convert_image_to_screen_data(image.resize(screen_dimensions))
            robot.screen.set_screen_with_image_data(screen_data, 10.0, True)

    def robot_moves(self):
        with anki_vector.Robot() as robot:
            robot.behavior.turn_in_place(degrees(-360))
            robot.behavior.turn_in_place(degrees(360))
            robot.behavior.drive_straight(distance_mm(80), speed_mmps(80))
            robot.behavior.turn_in_place(degrees(180))
            robot.behavior.drive_straight(distance_mm(80), speed_mmps(80))
            robot.behavior.turn_in_place(degrees(180))

    def remote_control(self):
        args = anki_vector.util.parse_command_args()
        with anki_vector.Robot(args.serial,
                               show_viewer=True,
                               show_3d_viewer=True,
                               enable_face_detection=True,
                               enable_custom_object_detection=True,
                               enable_nav_map_feed=True):
            print("Starting 3D Viewer. Use Ctrl+C to quit.")

            while True:
                time.sleep(100)


```
