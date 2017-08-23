#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
import tf

import math
from control_msgs.msg import _SingleJointPositionResult

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number

MAX_DECEL = 1.0
MAX_ACCEL = 1.0

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/obstacle_waypoints', PoseStamped, self.obstacle_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.velocity = rospy.get_param('~velocity')/3.6*0.27778 #in m/s 

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        
        self.lane = None
        
        self.current_pose = None
        
        self.traffic_waypoint = None
        
        rospy.loginfo('WaypointUpdater started')

        rospy.spin()

    def publish(self, waypoints):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)

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
        
    def get_direction(self, pose1, pose2):
        return math.atan2((pose2.position.y - pose1.position.y) , (pose2.position.x - pose1.position.x))
        
    def is_infront(self, wp_pose1, pose2):
        direction = self.get_direction(wp_pose1, pose2)
        
        return abs(direction) < math.pi*0.5 
        
    def pose_cb(self, pose):
        self.current_pose = pose
        yaw = self.get_yaw(pose.pose)
        
        if self.lane is None:
            return
        
        final_waypoints = []
        
        (start,dir) = self.get_next_waypoint(self.lane.waypoints, self.current_pose)
        
        rospy.loginfo('wp %s dir %s',start,dir)
        next_wp = start

        dist = self.distance_pose_to_pose(self.lane.waypoints[next_wp].pose.pose, self.current_pose.pose)
        dir = self.get_direction(self.lane.waypoints[next_wp].pose.pose, self.current_pose.pose)
        inFront = self.is_infront(self.lane.waypoints[next_wp].pose.pose, self.current_pose.pose)
        rospy.loginfo("wp %s dist: %s dir: %s ahead: %s",next_wp,dist,dir,inFront)
        
        for wp in range(LOOKAHEAD_WPS):

            if dir == True:
                next_wp = next_wp + 1 
            else:
                next_wp = next_wp - 1 
                
            p = Waypoint(self.lane.waypoints[next_wp].pose, self.lane.waypoints[next_wp].twist)
            #set a default speed in 
            p.twist.twist.linear.x = self.velocity          
            final_waypoints.append(p)
        
#        if self.traffic_waypoint is not None and self.current_pose is not None: 
#            distance = distance_pose_to_pose(traffic_waypoint.pose, current_pose)            
#            direction = get_direction(self.current_pose, self.traffic_waypoint.pose)
            
#            is_in_front = direction < math.pi*0.5 and direction > (-math.pi*0.5) 
#            if is_in_front and distance < 30:
#                rospy.loginfo('need to stop. distance:',distance)
#                decelerate(waypoints);
#            if not is_in_front:
#                self.traffic_waypoint = None
#                rospy.logdebug('traffic waypoint passed')
                
            
        rospy.loginfo('published final_waypoints')
        self.publish(final_waypoints)
        
#        rospy.loginfo('Current pose received:%s yaw: %s',pose, yaw)

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
        
    def waypoints_cb(self, lane):
        rospy.loginfo('Waypoints received %s',len(lane.waypoints))
        
        self.lane = lane
        
              
    def accelerate(self, waypoints):
        rospy.logdebug('accelerate')
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:-1][::-1]:
            dist = self.distance(wp.pose.pose.position, last.pose.pose.position)            
            vel = math.sqrt(2 * MAX_ACCEL * dist) * 3.6
            if vel > 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            rospy.logdebug('accelerate:',wp.twist.twist.linear.x)
        return waypoints
        
    def decelerate(self, waypoints):
        rospy.logdebug('decelerate')
        last = waypoints[-1]
        last.twist.twist.linear.x = 0.
        for wp in waypoints[:-1][::-1]:
            dist = self.distance(wp.pose.pose.position, last.pose.pose.position)
            vel = math.sqrt(2 * MAX_DECEL * dist) * 3.6
            if vel < 1.:
                vel = 0.
            wp.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            rospy.logdebug('decelerate:',wp.twist.twist.linear.x)
        return waypoints
        
    def traffic_cb(self, traffic_waypoint):
        self.traffic_waypoint = traffic_waypoint
        rospy.loginfo('Traffic waypoint received:%s',traffic_waypoint)
        
    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance_pose_to_pose(self, pose1, pose2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        dist = dl(pose1.position, pose2.position)
        return dist
        
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
