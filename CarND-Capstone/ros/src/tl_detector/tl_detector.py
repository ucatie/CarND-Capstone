#!/usr/bin/env python
import rospy
import os
import image_geometry
from std_msgs.msg import Int32
from tf2_msgs.msg import TFMessage
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
import time
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

        # first waypoint index at the previous iteration
        self.prev_first_wpt_index = 0
        
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

        self.traffic_light_is_close = rospy.get_param('~traffic_light_is_close', 50)
        rospy.loginfo("traffic_light_is_close:%f",self.traffic_light_is_close)
                
        self.is_simulator = rospy.get_param('~is_simulator', True)
        rospy.loginfo("is_simulator:%s",self.is_simulator)
        
        self.SVC_PATH =  rospy.get_param('~SVC_PATH','svc.p')       
        rospy.loginfo("SVC_PATH:%s",self.SVC_PATH)
        
        self.FCN_PATH =  rospy.get_param('~FCN_PATH','fcn')       
        rospy.loginfo("FCN_PATH:%s",self.FCN_PATH)

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = None
        if self.create_ground_truth:
            sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
            
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
#        sub7 = rospy.Subscriber('/tf', TFMessage, self.tf_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.upcoming_traffic_light_pub = rospy.Publisher('/traffic_light', TrafficLight, queue_size=1)
        #puplish the sub images for testing, can be viewed in rviz tool 
        self.upcoming_traffic_light_image_pub = rospy.Publisher('/traffic_light_image', Image, queue_size=1)

        
        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.SVC_PATH,self.is_simulator,self.FCN_PATH)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
        self.has_image = False
        self.green_to_yellow_or_red = False       
        self.transform = None         

        self.loop()

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
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
#        rospy.loginfo('image received')
        
        self.has_image = True
        self.camera_image = msg
        
    def loop(self):
        rate = rospy.Rate(5) # 10Hz
        while not rospy.is_shutdown():
        
            light_wp, state = self.process_traffic_lights()
            if state == None:
                continue
     
            '''
            Publish upcoming red lights at camera frequency.
            Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
            of times till we start using it. Otherwise the previous stable state is
            used.
            '''
            if self.state != state:
                self.state_count = 0
                self.state = state
                rospy.logdebug('state change %s',state) 
                
            if self.state_count >= STATE_COUNT_THRESHOLD:

                if state == TrafficLight.YELLOW and self.last_state == TrafficLight.GREEN:
                    self.green_to_yellow_or_red = True
                elif state == TrafficLight.RED:
                    self.green_to_yellow_or_red = True
                elif state == TrafficLight.GREEN:
                    self.green_to_yellow_or_red = False
                else:
                    self.green_to_yellow_or_red = False
                
                self.last_state = self.state
               
                if self.green_to_yellow_or_red:
                    rospy.logdebug('green_to_yellow_or_red %s %s',light_wp,state)
                else:
                    light_wp = -1
 #               rospy.loginfo('light_wp %s %s',light_wp,state)
                
                self.last_wp = light_wp
#                rospy.loginfo('state pub1 %s',self.last_wp) 
                self.upcoming_red_light_pub.publish(int(light_wp))
            else:
                self.upcoming_red_light_pub.publish(int(self.last_wp))
#                rospy.loginfo('state pub2 %s',self.last_wp) 
            self.state_count += 1
            rate.sleep()
    
    def distance_pose_to_pose(self, pose1, pose2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)
        dist = dl(pose1.position, pose2.position)
        return dist
    
    def get_direction(self, pose1, pose2):
        return math.atan2((pose2.position.y - pose1.position.y) , (pose2.position.x - pose1.position.x))
        
    def is_infront(self, wp_pose1, pose2):
        direction = self.get_direction(wp_pose1, pose2)        
        return abs(direction) < math.pi*0.5 
        		
    def get_yaw(self, pose1):
        
        quaternion = (
            pose1.orientation.x,
            pose1.orientation.y,
            pose1.orientation.z,
            pose1.orientation.w)
        euler = tf.transformations.euler_from_quaternion(quaternion)
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]    
        return yaw
                        
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
            self.listener.waitForTransform("base_link",
                  "world", now, rospy.Duration(0.02))
#            (trans, rot) = self.listener.lookupTransform("base_link",
#                  "world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

#        rospy.loginfo('base_link received trans %s rot %s',trans,rot)
        
        camera_point=PointStamped()
        camera_point.header.frame_id = "/world"
        camera_point.header.stamp =rospy.Time(0)
        camera_point.point.x = point_in_world.x
        camera_point.point.y = point_in_world.y
        camera_point.point.z = point_in_world.z
        p = self.listener.transformPoint("base_link",camera_point)
        
        #correct shift of traffic light to left / right because of car heading         
        car_yaw = self.get_yaw(self.current_pose.pose)
                
        y_offset = p.point.x*math.sin(car_yaw)
        
        #https://en.wikipedia.org/wiki/Pinhole_camera_model#The_geometry_and_mathematics_of_the_pinhole_camera        
        x = -(p.point.y + y_offset)/ p.point.x * fx + image_width*0.5
        #experiments showed that there are 62 pixel offset of the camera 
        y = 62 + image_height - (p.point.z / p.point.x * fy + image_height*0.5) 
        	
#        rospy.loginfo('3D map (%s %s) camera (%s %s %s) pixel (%s %s)',point_in_world.x,point_in_world.y,p.point.x, p.point.y, p.point.z, x,y)
		
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

        image_age = time.time() - self.camera_image.header.stamp.secs-(self.camera_image.header.stamp.nsecs/100000000)
        if self.camera_image.header.stamp.secs > 0 and image_age > 0.1:
            rospy.logdebug("image message delay %s %s %s",time.time(),self.camera_image.header.stamp,image_age)
            
        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

#        x, y = self.project_to_image_plane(world_light.pose.position)
		
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']
		
        #use light location to zoom in on traffic light in image
        	
        shape = cv_image.shape
        if (shape[0] != image_height or shape[1] !=  image_width):
            cv_image = cv2.resize(cv_image, (image_height, image_width), interpolation = cv2.INTER_AREA)
#            rospy.loginfo("resize %s %s ", shape, (image_height, image_width))
            
        rgbimage = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        (x,y) = self.light_classifier.find_classification(rgbimage)
        if x is None:
            return TrafficLight.UNKNOWN

        #outdside image
        y = min(y,image_height-32)
        x = max(min(x,image_width-32),32)
       
        x1 = x-32 
        y1 = y-32
        x2 = x+32 
        y2 = y+32
        
        if not self.is_simulator:
            x1 = x-32 
            y1 = y-64
            x2 = x+32 
            y2 = y+64
                        
        region = cv_image[y1:y2, x1:x2]
        if not self.is_simulator:
            region = cv2.resize(region, (128, 128), interpolation = cv2.INTER_AREA)
            
#        rospy.loginfo('region %s %s %s %s org: %s region:%s',x1,y1,x2,y2, rgbimage.shape, region.shape)

#        traffic_image = self.bridge.cv2_to_imgmsg(region, "bgr8")
#        self.upcoming_traffic_light_image_pub.publish(traffic_image);     
#        rospy.loginfo('traffic light image published')

        #Get ground truth classification and save it as part of the image name 
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

        region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        return self.light_classifier.get_classification(region)
        
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        world_light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.current_pose is None):
            return -1, None
        
        min_distance = 9999.0
        light_wp = -1

        #transform fast avoiding wait cycles
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("base_link","world", now, rospy.Duration(0.02))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
#            rospy.logwarn("Failed to find camera to map transform 0.02 duration")
            try:
                self.listener.waitForTransform("base_link","world", now, rospy.Duration(0.1))
            except (tf.Exception, tf.LookupException, tf.ConnectivityException):
                rospy.logwarn("Failed to find camera to map transform")
                return -1, None

#        rospy.loginfo('base_link received trans %s rot %s',trans,rot)
        
        wtl=PointStamped()
        wtl.header.frame_id = "/world"
        wtl.header.stamp =rospy.Time(0)
        wtl.point.z = 0
        for wp in range(len(stop_line_positions)):
            wtl.point.x = stop_line_positions[wp][0]
            wtl.point.y = stop_line_positions[wp][1]
            # Transform first waypoint to car coordinates
            ctl = self.listener.transformPoint("base_link",wtl)
            pose = PoseStamped()
            pose.pose.position.x = stop_line_positions[wp][0]
            pose.pose.position.y = stop_line_positions[wp][1]
            pose.pose.position.z = 0
           
            #only points ahead  
            if ctl.point.x > 0 and ctl.point.x < min_distance and abs(ctl.point.y) < 10:
                min_distance = ctl.point.x 
                world_light = pose
                light_wp = wp

        #nothing ahead
        if world_light is None:
            return -1, TrafficLight.UNKNOWN        
#        rospy.loginfo('stop line distance: %s pose %s', min_distance,(pose.pose.position.x,pose.pose.position.y)) 
        
        if min_distance < self.traffic_light_is_close and min_distance >=0:
            rospy.logdebug('stop line close: %s dir %s', min_distance,dir) 
        else:
            return -1, TrafficLight.UNKNOWN

        #TODO find the closest visible traffic light (if one exists)
        if world_light is not None:
            state = self.get_light_state(world_light, min_distance)
            
            header = std_msgs.msg.Header()
            header.frame_id = 'world'
            header.stamp = rospy.Time.now()
#            self.upcoming_traffic_light_pub.publish(TrafficLight(header,world_light,state))

            # Iterate through the complete set of waypoints until we found the closest
            first_wpt_index = -1
            min_wpt_distance = float('inf')
            distance_decreased = False
            for index, waypoint in enumerate(self.lane.waypoints[self.prev_first_wpt_index:] + self.lane.waypoints[:self.prev_first_wpt_index], start=self.prev_first_wpt_index):
                current_wpt_distance = math.sqrt((waypoint.pose.pose.position.x-stop_line_positions[light_wp][0])**2 + (waypoint.pose.pose.position.y-stop_line_positions[light_wp][1])**2)
                if distance_decreased and current_wpt_distance > min_wpt_distance:
                    break
                if current_wpt_distance > 0 and current_wpt_distance < min_wpt_distance:
                    min_wpt_distance = current_wpt_distance
                    first_wpt_index = index
                    distance_decreased = True
            first_wpt_index %= len(self.lane.waypoints)
            self.prev_first_wpt_index = first_wpt_index - 1
            
            return first_wpt_index, state
        
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')


