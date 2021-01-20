import os
import sys
import numpy as np
import time
from PIL import Image
import anki_vector
from anki_vector.util import degrees
import anki_vector.camera
from anki_vector.util import degrees, distance_mm, speed_mmps
import anki_vector
import numpy as np

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
import cv2
import argparse
from keras_yolo3.yolo import YOLO
from PIL import Image
from timeit import default_timer as timer
from utils import load_extractor_model, load_features, parse_input, detect_object, detect_frame
import pandas as pd
from Train_Utils import get_anchors

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
class_list=['Computer monitor', 'Mobile phone', 'Car']
# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "yolo.h5")
model_classes = os.path.join(model_folder, "data_coco.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

FLAGS = None


def get_classnames(classes):
    try:
        class_list = []
        for class_names in classes:
           class_list.append(class_names)
        return ', '.join(class_list)

    except:
        return 'no objects'


if __name__ == "__main__":
    # Delete all default flags
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    """
    Command line options
    """

    parser.add_argument(
        "--no_save_img",
        default=False,
        action="store_true",
        help="Only save bounding box coordinates but do not save output images with annotated boxes. Default is False.",
    )

    parser.add_argument(
        "--file_types",
        "--names-list",
        nargs="*",
        default=[],
        help="Specify list of file types to include. Default is --file_types .jpg .jpeg .png .mp4",
    )

    parser.add_argument(
        "--yolo_model",
        type=str,
        dest="model_path",
        default=model_weights,
        help="Path to pre-trained weight files. Default is " + model_weights,
    )

    parser.add_argument(
        "--anchors",
        type=str,
        dest="anchors_path",
        default=anchors_path,
        help="Path to YOLO anchors. Default is " + anchors_path,
    )

    parser.add_argument(
        "--classes",
        type=str,
        dest="classes_path",
        default=model_classes,
        help="Path to YOLO class specifications. Default is " + model_classes,
    )

    parser.add_argument(
        "--gpu_num", type=int, default=1, help="Number of GPU to use. Default is 1"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        dest="score",
        default=0.25,
        help="Threshold for YOLO object confidence score to show predictions. Default is 0.25.",
    )


    FLAGS = parser.parse_args()

    # Split images and videos
    img_endings = (".jpg", ".jpeg", ".png")
    vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")

    anchors = get_anchors(anchors_path)
    # define YOLO detector
    yolo = YOLO(
        **{
            "model_path": FLAGS.model_path,
            "anchors_path": anchors_path,
            "classes_path": FLAGS.classes_path,
            "score": FLAGS.score,
            "gpu_num": FLAGS.gpu_num,
            "model_image_size": (416, 416),
        }
    )

    # labels to draw on images
    class_file = open(FLAGS.classes_path, "r")
    input_labels = [line.rstrip("\n") for line in class_file.readlines()]
    print("Found {} input labels: {} ...".format(len(input_labels), input_labels))
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color1 = (0, 0, 255)
    thickness = 1
    with anki_vector.Robot() as robot:
        start = timer()
        while True:
            robot.camera.init_camera_feed()
            img = cv2.cvtColor(np.array(robot.camera.latest_image.raw_image), cv2.COLOR_RGB2BGR)
            img2 = Image.fromarray(img)
            bbox, class_name, score = detect_frame(
                yolo,
                img2
            )
            c = cv2.waitKey(1)
            if c == 27:
                break
            if len(bbox)!=0:
                color = list(np.random.random(size=3) * 256)
                for i in range(len(bbox)):
                    cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), color, 2)
                    cv2.putText(img,  input_labels[class_name[i]]+'  Prob:'+str(score[i]), (bbox[i][0],bbox[i][1]), font,
                                fontScale, color, thickness, cv2.LINE_AA)

            cv2.imshow('output', img)
        cv2.destroyAllWindows()
        end = timer()

    yolo.close_session()
