This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. For more information about the project, see the project introduction [here](https://classroom.udacity.com/nanodegrees/nd013/parts/6047fe34-d93c-4f50-8336-b70ef10cb4b2/modules/e1a23b06-329a-4684-a717-ad476f0d8dff/lessons/462c933d-9f24-42d3-8bdc-a08a5fc866e4/concepts/5ab4b122-83e6-436d-850f-9f4d26627fd9).

### Installation 

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop). 
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space
  
  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/self-driving-car-sim/releases/tag/v0.1).

### Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/carnd_capstone.git
```

2. Install python dependencies
```bash
cd carnd_capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator
```bash
unzip lights_no_cars.zip
cd lights_no_cars
chmod +x ros_test.x86_64
./ros_test.x86_64
```

### Team Members

| Name                | Email                             | Slack        | TimeZone |
|:-------------------:|:---------------------------------:|:------------:|:--------:|
| Frank Schneider     | frank@schneider-ip.de             | @fsc         | UTC+2 |
| Sebastian Nietardt  | sebastian.nietardt@googlemail.com | @sebastian_n | UTC+2 |
| Sebastiano Di Paola | sebastiano.dipaola@gmail.com      | @abes975	 | UTC+2 |
| Juan Pedro Hidalgo  | juanpedro.hidalgo@hotmail.com     | @jphidalgo   | UTC+2 |
| Ramiz Raja          | informramiz@gmail.com             | @ramiz       | UTC+5 |

### task break down table

task break down and management is done at:

[https://waffle.io/ucatie/CarND-Capstone](https://waffle.io/ucatie/CarND-Capstone)

### Node description

#### way point updater
Publishes 200 final_waypoints. Dummy velocity set, periodically update seems to be buggy. 
Subscribed to obstacle_waypoints. 
Subscribed to base_waypoints. Callback stores lane message
Subscribed to traffic waypoint. Callback stores traffic waypoint message
Subscribed to current_pose. Callback publishes final waypoint list having 200 wp ahead.

#### twist_controller
simple default implementation for dummy behaviour

#### yaw_controller

#### DBWNode
sending directly throttle and steering command. No PID yet

#### tl_detector
Subscribed to base_waypoints. Callback stores lane message
Subscribed to current_pose. Callback publishes final waypoint list having 200 wp ahead.
Subscribed to vehicle/traffic_lights. 
Subscribed camera/image_raw.
Publishes cropped images of traffic lights for testing on 'traffic_light_image'
Publishes upcoming traffic light (TrafficLight) on '/traffic_light'
Publishes upcoming red traffic light waypoint to '/traffic_waypoint' (way point index)

parameters are used to define to create ground truth and / or training data. Please check launch file

#### tl_detector_train
node to train a svc for four states (red, green, yellow, unknown'.data is read as defined by parameters. Should be the same as tl_detector parameters. Parameter task allows different operations on the svm training.

rosrun tl_detector tl_detector_train.py

if task "best" is executed a trained svc.p file is written, which could be used in the tl_classifier.
Using HSV, All channels, Histogram only for feature space I got 93% accuracy on red 1290 green 213 yellow 203 unknown 56
images.

might require installation of these packages: 
sudo pip install -U scikit-learn
sudo pip install -U scikit-image
sudo pip install -U Pillow
sudo pip install -U matplotlib

#### tl_classifier
node to classify images using the trained svc.
parameters are used to define the svc model file.

#### tl_classifier_test
node to classify images of the data_gt or data_test folder. Uses the trained svc and calls tl_classifier code.
parameters are used to define the data folder and svc model file.

roslaunch tl_detector test_classifier.launch 

#### helper tool  


