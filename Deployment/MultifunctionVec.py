import os
import sys
import time
from anki_vector.util import degrees
import anki_vector.camera
from anki_vector.util import degrees, distance_mm, speed_mmps
import anki_vector
import numpy as np
import collections
import cv2
import speech_recognition as sr
from keras_yolo3.yolo import YOLO
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object, detect_frame
from Train_Utils import get_anchors

screen_dimensions = anki_vector.screen.SCREEN_WIDTH, anki_vector.screen.SCREEN_HEIGHT


def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

data_folder = os.path.join(get_parent_dir(n=1), "Data")
model_folder = os.path.join(data_folder, "Model_Weights")
model_weights = os.path.join(model_folder, "yolo.h5")
model_classes = os.path.join(model_folder, "data_coco.txt")
anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")
robot = anki_vector.Robot(anki_vector.util.parse_command_args().serial)

yolo = YOLO(
            **{
                "model_path": model_weights,
                "anchors_path": anchors_path,
                "classes_path": model_classes,
                "score": 0.25,
                "gpu_num": 1,
                "model_image_size": (416, 416),
            }
        )


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


if __name__ == "__main__":
    vec = VecRobot()
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    vec.vector_speaks('Hey What you want me to do?')
    robot = anki_vector.Robot()
    while True:

        for j in range(100):
            print('Guess {}. Speak!')
            guess = vec.speech_regognizer(recognizer, microphone)
            if guess["listened"]:
                break
            if not guess["success"]:
                break

        print("You said: {}".format(guess["listened"]))
        if 'room' in guess["listened"]:
            classes = vec.real_time_detection(param=None)
            text = vec.get_classnames(classes)
            vec.vector_speaks('I have detected {}'.format(text))

        if 'car' in guess["listened"]:
            classes = vec.real_time_detection('car')
            if 'car' in classes:
                with anki_vector.Robot() as robot:
                    robot.behavior.drive_straight(distance_mm(200), speed_mmps(80))
                    vec.vector_speaks('I Found your car')

        if 'crazy steps' in guess["listened"]:
            vec.robot_moves()

        if 'angry' in guess["listened"]:
            vec.display_image('red2.png')
            vec.vector_speaks('This was my devil Lucifer eyes')
            vec.display_image('red3.png')
            vec.vector_speaks('And it was my angry eyes')

        if 'love you' in guess["listened"]:
            vec.vector_speaks('I love you too')

        if 'come back' in guess["listened"]:
            with anki_vector.Robot() as robot:
                robot.behavior.drive_straight(distance_mm(-100), speed_mmps(80))
                robot.behavior.turn_in_place(degrees(180))

        if 'come out' in guess["listened"]:
            vec.obstacle_detection()
