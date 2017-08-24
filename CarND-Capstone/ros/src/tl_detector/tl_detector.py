#!/usr/bin/env python
import rospy
import image_geometry
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, PointStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from traffic_light_config import config
import tf
import cv2
import math

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.current_pose = None
        self.lane = None
        self.camera_image = None
        self.lights = []
        self.traffic_light_is_close = rospy.get_param('~traffic_light_is_close', 50)
        
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights helps you acquire an accurate ground truth data source for the traffic light
        classifier, providing the location and current color state of all traffic lights in the
        simulator. This state can be used to generate classified images or subbed into your solution to
        help you work on another single component of the node. This topic won't be available when
        testing your solution in real life so don't rely on it in the final submission.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/camera/image_raw', Image, self.image_cb)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.upcoming_traffic_light_pub = rospy.Publisher('/traffic_light', Image, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0
		
		#https://en.wikipedia.org/wiki/Pinhole_camera_model#The_geometry_and_mathematics_of_the_pinhole_camera		
        self.cam_model = image_geometry.PinholeCameraModel()		
        self.config = config
        ci = self.config.camera_info
        rospy.loginfo('config data available')
        # Intrinsic camera matrix for the raw (distorted) images.
        #     [fx  0 cx]
        # K = [ 0 fy cy]
        #     [ 0  0  1]		
        k = [[ ci.focal_length_x,      0,          ci.image_width*0.5],
             [        0,            ci.focal_length_x, ci.image_height*0.5],
             [        0,                  0,                   1         ]]
        
        # Projection/camera matrix
        #     [fx'  0  cx' Tx]
        # P = [ 0  fy' cy' Ty]
        #     [ 0   0   1   0]
        # For monocular cameras, Tx = Ty = 0. 
        p = [[ ci.focal_length_x,      0,             ci.image_width*0.5,      0],
             [        0,            ci.focal_length_x, ci.image_height*0.5,    0],
             [        0,                  0,                    1,             0]]
        
        # 3x3 row-major matrix		
        r = [[ 1,0,0],
        	 [0,1,0],
        	 [0,0,1]]#no rotation
        
        cam_info_msg = CameraInfo()
        cam_info_msg.header.frame_id = 'camera'
        cam_info_msg.width = ci.image_width
        cam_info_msg.height = ci.image_height
        cam_info_msg.distortion_model ='plumb_bob'
        cam_info_msg.D = [0,0,0,0,0]#no distortion
        cam_info_msg.K = k
        cam_info_msg.P = p
        cam_info_msg.R = r
        cam_info_msg.binning_x=1
        cam_info_msg.binning_y=1
        
        self.cam_model.fromCameraInfo( cam_info_msg)       
        rospy.loginfo('camera info %s',cam_info_msg)
        
        rospy.spin()
        
    def pose_cb(self, msg):
#        rospy.loginfo('Pose received')
        self.current_pose = msg

    def waypoints_cb(self, lane):
#        rospy.loginfo('Waypoints received')
        
        self.lane = lane

    def traffic_cb(self, msg):
#        rospy.loginfo('traffic light received')
        self.lights = msg.lights

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
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
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
 		
    def project_to_image_plane(self, point_in_world, point_in_car):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = config.camera_info.focal_length_x
        fy = config.camera_info.focal_length_y

        image_width = config.camera_info.image_width
        image_height = config.camera_info.image_height

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
        camera_point.point.x=point_in_world.x
        camera_point.point.y=point_in_world.y
        camera_point.point.z=point_in_world.z
        p = self.listener.transformPoint("/base_link",camera_point)
        
        (x,y) = self.cam_model.project3dToPixel((p.point.x, p.point.y, p.point.z))	
#        rospy.loginfo('3D map (%s %s) camera (%s %s %s) pixel (%s %s)',point_in_world.x,point_in_world.y,p.point.x, p.point.y, p.point.z, x,y)
		
        return (int(x), int(y))

    def get_light_state(self, world_light, car_light):
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

        x, y = self.project_to_image_plane(world_light.pose.position, car_light.pose.position)
		
        image_width = config.camera_info.image_width
        image_height = config.camera_info.image_height
		
              #use light location to zoom in on traffic light in image
        x1 = x-128
        y1 = y-128
        x2 = x+128
        y2 = y+128
        if x1 < 0 or x2 > image_width or y1 < 0 or y2 > image_height:
        	rospy.loginfo('outside image %s',world_light.pose.position)
        	return TrafficLight.UNKNOWN
        	
        resized = cv2.resize(cv_image, (image_height, image_width,), interpolation = cv2.INTER_AREA)            
        region = resized[x1:x2, y1:y2]
#        rospy.loginfo('region %s %s %s %s org: %s region:%s',x1,y1,x2,y2, resized.shape, region.shape)
        
        self.camera_image.encoding = "rgb8"
        traffic_image = self.bridge.cv2_to_imgmsg(region, "bgr8")
        
        self.upcoming_traffic_light_pub.publish(traffic_image);
#        rospy.loginfo('traffic light image published')
        
        
        #Get classification
        return self.light_classifier.get_classification(cv_image)
        
    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        light_positions = config.light_positions
        if(self.current_pose is None):
            return -1, TrafficLight.UNKNOWN
        
        min_distance = 9999.0
        light = None
        for wp in range(len(light_positions)):
            pose = PoseStamped()
            pose.pose.position.x = self.config.light_positions[wp][0]
            pose.pose.position.y = self.config.light_positions[wp][1]
            dist = self.distance_pose_to_pose(self.current_pose.pose, pose.pose)
#            rospy.loginfo('traffic light: %s %s %s', pose.pose.position.x, pose.pose.position.y, dist)
             
            if dist < min_distance:
                min_distance = dist 
                light = pose

        dir = self.get_direction(self.current_pose.pose,light.pose)
        if abs(dir) < math.pi*0.5 and min_distance < self.traffic_light_is_close:
            rospy.loginfo('traffic light close: %s', min_distance) 
        else:
            return -1, TrafficLight.UNKNOWN

        #TODO find the closest visible traffic light (if one exists)
        
        if light is not None:
            x = min_distance*math.cos(dir)
            y = min_distance*math.sin(dir)
            car_pose = PoseStamped()
            car_pose.pose.position.x = x
            car_pose.pose.position.y = y
            car_pose.pose.position.z = 0
#            rospy.loginfo('traffic light local: %s %s', x,y) 
            state = self.get_light_state(light, car_pose)
            return light, state
        
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
