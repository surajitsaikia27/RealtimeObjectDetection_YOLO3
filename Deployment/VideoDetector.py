import os
import sys
import numpy as np
import glob
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
from keras_yolo3.yolo import YOLO, detect_video
import pandas as pd
from Train_Utils import get_anchors

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "yolo.h5")
model_classes = os.path.join(model_folder, "data_coco.txt")
class_list=model_classes
print(class_list,len(class_list))
anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")
print()
FLAGS = None
color = list(np.random.random(size=3) * 256)
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
        default=0.70,
        help="Threshold for YOLO object confidence score to show predictions. Default is 0.25.",
    )
    parser.add_argument(
        "--video",
        type=str,
        dest="video_path",
        default='./carvid.mp4',
        help="Path to the videos",
    )
    parser.add_argument(
        "--video_out",
        type=str,
        dest="out_path",
        default='./vidout/out.avi',
        help="Path to the videos",
    )


    FLAGS = parser.parse_args()

    # Split images and videos
    img_endings = (".jpg", ".jpeg", ".png")
    vid_endings = (".mp4", ".mpeg", ".mpg", ".avi")


    anchors = get_anchors(anchors_path)

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
    FLAGS.output='vidout'

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    thickness = 1

    Path = FLAGS.video_path
    out_path=FLAGS.out_path


    if True:
        start = timer()

        cap = cv2.VideoCapture(Path)
        cap.set(cv2.CAP_PROP_FPS, 20)
        img_array=[]
        while True:
            try:
                ret, img = cap.read()

                img2 = Image.fromarray(img)
                bbox, class_name, score = detect_frame(
                    yolo,
                    img2
                )

                c = cv2.waitKey(1)
                if c == 27:
                    break

                if len(bbox) != 0:
                    for i in range(len(bbox)):
                        sc ="{:.2f}".format(score[i])
                        G= (class_name[i]*45)%55
                        B= (class_name[i]*500)%255
                        R = (class_name[i] * 255) % 45
                        color = list(np.random.random(size=3) * 256)
                        cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), (int(R), 255,int(B)), 2)
                        cv2.putText(img, input_labels[class_name[i]] + '  Prob:' + str(sc), (bbox[i][0], bbox[i][1]),
                                    font, fontScale, (int(R),255, int(B)), thickness, cv2.LINE_AA)

                cv2.imshow('Input', img)
                img_array.append(img)


            except:
                AttributeError
                break
        cap.release()
        cv2.destroyAllWindows()
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, (640, 360))
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()
        out.release()

    yolo.close_session()

