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

| task                               | module            |owner         | status      | description                                                         |
|:----------------------------------:|:-----------------:|:------------:|:-----------:|:-------------------------------------------------------------------:|
|create waypoint list                | way point updater |              | started     | list of 200 points ahead published having correct target velocity|
|react on traffic waypoint event     | way point updater |              | started     | if a traffic waypoint has been received and its close, car has to stop. Waypoint list target velocities have to reflect slow done until the car stops in front of the traffic light. When to accelerate again? When the traffic waypoint message is not received anymore?|
|react on obstacle waypoint event    | way point updater |              | not started | if a obstacle waypoint has been received and its close, car has to stop. |
|create break and throttle commands  | twist_controller  |              | not started | use pid and lowpass or own pid code to smoothly accelerate and break following subscribed twist messages.|
|create steering commands            | yaw_controller    |              | not started | create smooth steering commands by converting target linear and angular velocity following subscribed twist messages.|
|Traffic Light Detection             | tl_classifier     |              | not started | create a FCN classifier similar to Scene Segmentation to detect traffic lights |
|Obstacle Detection                  | obstacle_classifier|             | not started | requirement unclear |
|create TL ground thruth             | helper tool       |              | not started | let the car drive around  and record ground truth images required to train the classifier. |
|create training data TL Detection   | helper tool       |              | not started | let the car drive around  and record train images required to train the classifier. |
|train TL Detection                  | tl_classifier     |              | not started | create a FCN classifier similar to Scene Segmentation to detect traffic lights |
|                                    |                   |              |             |                                                                       |
|                                    |                   |              |             |                                                                       |
|                                    |                   |              |             |                                                                       |
|                                    |                   |              |             |                                                                       |

### Module stories, issues, questions or open points

#### way point updater
publishing 200 final_waypoints. Dummy velocity set, periodically update seems to be buggy. 
Subscribed to obstacle_waypoints. No callback
Subscribed to base_waypoints. Callback stores lane message
Subscribed to traffic waypoint. Callback stores traffic waypoint message
Subscribed to current_pose. Callback publishes final waypoint list having 200 wp ahead.
Next steps: test call back current pose. Set velocity. React on traffic message

#### twist_controller

#### yaw_controller

#### DBWNode
sending only dummy throttle command

#### tl_classifier

#### helper tool  


