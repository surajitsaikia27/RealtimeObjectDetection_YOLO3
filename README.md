# RealtimeObjectDetection_YOLOv3: Creating a Realtime custom object Detector

Using this Repository you can build an object detector that can detect objects in real time based on webcam feed.
To train the detector we will be using Google open Image Dataset (https://opensource.google/projects/open-images-dataset).

This repository has been tested using TensorFlow > 2 and Cuda 11.

### Download the Dataset for training

At First we need to download the dataset for training. 
Please refer to the following link to know how to download the dataset and annotate it.
https://surjeetsaikia.medium.com/build-your-realtime-custom-object-detector-using-yolov3-f61af825153f


## Repo structure
+ [`Training`](/Training/): Scripts to train the YOLOv3 model
+ [`Deployment`](/Deployment/): Script to run the trained YOLO detector using webcam
+ [`Data`](/Data/): Input Data, Output Data, Model Weights and Results
+ [`Utils`](/Utils/): Utility scripts used by main scripts


### Installation


Clone or download this repo with:
```
cd RealtimeObjectDetection_YOLOv3/
```

#### Install dependencies [Windows, Mac or Linux]
```
pip install -r requirements.txt
```



## Start Training

To train your model, please type the following command from the root directory.

```
python ./Training/TrainerOID.py

```
 
**To make everything run smoothly it is highly recommended to keep the original folder structure of this repo!**

The TrainerOID.py file has consist of various command-line options which you can modify in order to change epochs, learning rate, batch size etc. 
To speed up training, it is recommended to use a **GPU with CUDA** support. 


## Testing
Once you have trained the model, you will be needing a webcam to see how well the objects are detected.
The trained model will be saved as trained_weights_ODI.h5 in Model_weights folder.

To test the model, please type the following command from the root directory.


```
python ./Deployment/videoRT.py

```

 
This repository  is modified from the following repository [github repo](https://github.com/AntonMu/TrainYourOwnYOLO)!
 
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
