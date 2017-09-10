#!/usr/bin/env python
import rospy
import os
import image_geometry
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import yaml
import tf
import cv2
import math
import std_msgs
import matplotlib.image as mpimg

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.current_pose = None
        self.lane = None
        self.camera_image = None
        self.gt_lights = []
        self.run_dir = rospy.get_param('/run_dir')
        rospy.loginfo("run_dir:%s",self.run_dir)
        
        self.create_ground_truth = rospy.get_param('~create_ground_truth', False)
        rospy.loginfo("create_ground_truth:%s",self.create_ground_truth)
               
        if self.create_ground_truth:               
            self.ground_truth_dir = os.path.join(self.run_dir, rospy.get_param('~ground_truth_dir'))
            rospy.loginfo("ground_truth_dir:%s",self.ground_truth_dir)
            
            self.ground_truth_start_number = rospy.get_param('~ground_truth_start_number', 1)
            rospy.loginfo("ground_truth_start_number:%s",self.ground_truth_start_number)
                
            if not os.path.exists(self.ground_truth_dir):
                os.makedirs(self.ground_truth_dir)        
                os.makedirs(os.path.join(self.ground_truth_dir,'0'))        
                os.makedirs(os.path.join(self.ground_truth_dir,'1'))        
                os.makedirs(os.path.join(self.ground_truth_dir,'2'))        
                os.makedirs(os.path.join(self.ground_truth_dir,'4'))        

        self.create_train_data = rospy.get_param('~create_train_data', False) 
        rospy.loginfo("create_train_data:%s",self.create_train_data)

        if self.create_train_data:               
            self.train_data_dir = os.path.join(self.run_dir,rospy.get_param('~train_data_dir'))
            rospy.loginfo("train_data_dir:%s",self.train_data_dir)
            
            self.train_data_start_number = rospy.get_param('~train_data_start_number', 1)
            rospy.loginfo("train_data_start_number:%s",self.train_data_start_number)
            
            if self.create_train_data == True and not os.path.exists(self.train_data_dir):
                os.makedirs(self.train_data_dir)        
                
        self.traffic_light_is_close = rospy.get_param('~traffic_light_is_close', 50)
        rospy.loginfo("traffic_light_is_close:%f",self.traffic_light_is_close)
                
        self.SVC_PATH =  os.path.join(self.run_dir,rospy.get_param('~SVC_PATH','svc.p'))       
        rospy.loginfo("SVC_PATH:%s",self.SVC_PATH)

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights helps you acquire an accurate ground truth data source for the traffic light
        classifier, providing the location and current color state of all traffic lights in the
        simulator. This state can be used to generate classified images or subbed into your solution to
        help you work on another single component of the node. This topic won't be available when
        testing your solution in real life so don't rely on it in the final submission.
        '''
        sub3 = None
        if self.create_ground_truth:
            sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
            
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.upcoming_traffic_light_pub = rospy.Publisher('/traffic_light', TrafficLight, queue_size=1)
        #puplish the sub images for testing, can be viewed in rviz tool 
        self.upcoming_traffic_light_image_pub = rospy.Publisher('/traffic_light_image', Image, queue_size=1)
        
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.SVC_PATH)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
		
        rospy.spin()
        
    def pose_cb(self, msg):
#        rospy.loginfo('Pose received')
        self.current_pose = msg

    def waypoints_cb(self, lane):
#        rospy.loginfo('Waypoints received')
        
        self.lane = lane

    def traffic_cb(self, msg):
#        rospy.loginfo('traffic light received')
        self.gt_lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
#        rospy.loginfo('image received')
        
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
 
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
            
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            
            if state != TrafficLight.RED:
                light_wp = -1
            
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(int(light_wp))
        else:
            self.upcoming_red_light_pub.publish(int(self.last_wp))
        self.state_count += 1

    def distance_pose_to_pose(self, pose1, pose2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        dist = dl(pose1.position, pose2.position)
        return dist
    
    def get_direction(self, pose1, pose2):
        return math.atan2((pose2.position.y - pose1.position.y) , (pose2.position.x - pose1.position.x))
        
    def is_infront(self, wp_pose1, pose2):
        direction = self.get_direction(wp_pose1, pose2)        
        return abs(direction) < math.pi*0.5 
        		
    #return the index and a flag to increase the index for way points ahead
    def get_next_waypoint(self, waypoints, position):
        
        next_wp = -1
        for wp in range(len(waypoints)-1):
            inFront1 = self.is_infront(waypoints[wp].pose.pose, position.pose)
            inFront2 = self.is_infront(waypoints[wp+1].pose.pose, position.pose)
            
#            rospy.loginfo('%s %s %s',wp, inFront1, inFront2)
            
            if inFront1 and not inFront2:
               return (wp,False)
           
            if inFront2 and not inFront1:
               return (wp+1,True)
 		
    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

#        rospy.loginfo('base_link received trans %s rot %s',trans,rot)
        
        camera_point=PointStamped()
        camera_point.header.frame_id = "/world"
        camera_point.header.stamp =rospy.Time(0)
        camera_point.point.x = point_in_world.x
        camera_point.point.y = point_in_world.y
        camera_point.point.z = point_in_world.z
        p = self.listener.transformPoint("/base_link",camera_point)
        
        #https://en.wikipedia.org/wiki/Pinhole_camera_model#The_geometry_and_mathematics_of_the_pinhole_camera        
        x = -p.point.y / p.point.x * fx + image_width*0.5
        #experiments showed that there are 62 pixel offset of the cam 
        y = 62 + image_height - (p.point.z / p.point.x * fy + image_height*0.5) 
        	
        rospy.loginfo('3D map (%s %s) camera (%s %s %s) pixel (%s %s)',point_in_world.x,point_in_world.y,p.point.x, p.point.y, p.point.z, x,y)
		
        return (int(x), int(y))

    def get_light_state(self, world_light, distance):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        self.camera_image.encoding = "rgb8"
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(world_light.pose.position)
		
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']
		
        #use light location to zoom in on traffic light in image
        x1 = x-128 
        y1 = y-128
        x2 = x+128 
        y2 = y+128
        if x1 < 0 or x2 > image_width or y1 < 0 or y2 > image_height:
        	rospy.loginfo('outside image %s',world_light.pose.position)
        	return TrafficLight.UNKNOWN
        	
        shape = cv_image.shape
        if shape[0] != image_height or shape[1] !=  image_width:
            cv_image = cv2.resize(cv_image, (image_height, image_width), interpolation = cv2.INTER_AREA)
#            rosspy("resize %s %s ", shape, (image_height, image_width))
            
        region = cv_image[y1:y2, x1:x2]
        rospy.loginfo('region %s %s %s %s org: %s region:%s',x1,y1,x2,y2, cv_image.shape, region.shape)
        
        self.camera_image.encoding = "rgb8"
        traffic_image = self.bridge.cv2_to_imgmsg(region, "bgr8")
        
        self.upcoming_traffic_light_image_pub.publish(traffic_image);
        
#        rospy.loginfo('traffic light image published')
        
        
        #Get ground truth classification and save it as part of the image name 
        #heads up the traffic light messages are received delayed, classifications are wrong at least the first 6 pictures
        #dont forget to delete these mismatched images!!!!!
        if self.create_ground_truth:
            state = TrafficLight.UNKNOWN
            for i in range(len(self.gt_lights)):
                dist = self.distance_pose_to_pose(self.gt_lights[i].pose.pose, world_light.pose)
                #correct mismatch of traffic light positions
                dist = math.fabs(dist - 24)
#                rospy.loginfo('gt traffic light state %s %s',dist, self.gt_lights[i].state)
                if dist < 1.0:
                    state = self.gt_lights[i].state
                    rospy.loginfo('gt traffic light state %s',state)
                    break
            #write the to sub folder using state info. easier to move, if the state has changed slower than the image has ben received
            gt_image_path = os.path.join(os.path.join(self.ground_truth_dir,'{0}'.format(state)),'{0}.jpg'.format(self.ground_truth_start_number))
            cv2.imwrite(gt_image_path, region)
            rospy.loginfo('saved gt data %s',gt_image_path)
            self.ground_truth_start_number = self.ground_truth_start_number + 1


        #save the image as training data 
        if self.create_train_data:
            train_image_path = os.path.join(self.train_data_dir,'{0}.jpg'.format(self.train_data_start_number))
            cv2.imwrite(train_image_path, region)
            rospy.loginfo('saved train data %s',train_image_path)
            self.train_data_start_number = self.train_data_start_number + 1
#            image = mpimg.imread(train_image_path)
#            self.light_classifier.get_classification(image)            
            
        rgbregion = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        return self.light_classifier.get_classification(rgbregion)
        
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        world_light = None
        light_positions = self.config['light_positions']
        if(self.current_pose is None):
            return -1, TrafficLight.UNKNOWN
        
        min_distance = 9999.0
        light_wp = -1
        for wp in range(len(light_positions)):
            pose = PoseStamped()

            pose.pose.position.x = light_positions[wp][0]
            pose.pose.position.y = light_positions[wp][1]
            pose.pose.position.z = 7
            dist = self.distance_pose_to_pose(self.current_pose.pose, pose.pose)
#            rospy.loginfo('traffic light: %s %s %s %s',self.gt_lights[wp].header.frame_id, pose.pose.position.x, pose.pose.position.y, dist)
             
            if dist < min_distance:
                min_distance = dist 
                world_light = pose
                light_wp = wp

        dir = self.get_direction(self.current_pose.pose,world_light.pose)
        
        #get the orientation of the car
        quaternion = (
            self.current_pose.pose.orientation.x,
            self.current_pose.pose.orientation.y,
            self.current_pose.pose.orientation.z,
            self.current_pose.pose.orientation.w)        
        euler = tf.transformations.euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]        
        #add orientation pf car !!!
        if abs(dir) < math.pi*0.5 and min_distance < self.traffic_light_is_close:
            rospy.loginfo('traffic light close: %s dir %s car yaw %s', min_distance,dir,yaw) 
        else:
            return -1, TrafficLight.UNKNOWN

        #TODO find the closest visible traffic light (if one exists)
        
        if world_light is not None:
            state = self.get_light_state(world_light, min_distance)
            
            header = std_msgs.msg.Header()
            header.frame_id = '/world'
            header.stamp = rospy.Time.now()
            self.upcoming_traffic_light_pub.publish(TrafficLight(header,world_light,state))
            
            return light_wp, state
        
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
