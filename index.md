## RoboVector

In this project, a home robot known as Vector is programmed to do few tasks based on voiced commands. It is uses deep learning, computer vision and speech recognision, and is programmed using the Anki Vector SDK. The Vector SDK gives access to various capabilities of this robot, such as computer vision, Artificial intelligence, navigation and etc. You can design your own programs to make this robot pet imbibed with AI capabilities. Here, in this project, I have trained and used an real time object detector which lets the robot to recognise objects in its surrounding environment. Moreover, in this module, instruction are provided how to create your own customize object detector. However, in case if you dont have a Vector, then using this repository you can create your own custom object detector which can be tested using webcam.



### What Vector can do?
Vector is home companion robot powered by AI and can do certain basic functions based on voice commands. It can take your photos, show you weather, set timer, and many more. Also, since Vector comes with a SDK, one can program it do more stuffs than it normally does. So, in this project I gave vector the capabilty of real-time object detection along with some more other functionalities.

Following is the class blue print representing the functionalities of the vector.

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

    def vector_speaks(self, text):
      
    def speech_regognizer(self, recognizer, microphone):
    
    def obstacle_detection(self):
        
    def real_time_detection(self, param=None):
       
    def display_image(self, file_name):
      
    def robot_moves(self):
      
    def remote_control(self):
      
      
```
Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/surajitsaikia27/RoboVec/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
