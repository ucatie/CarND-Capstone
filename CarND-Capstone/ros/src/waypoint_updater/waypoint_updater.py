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

        #self.current_pose = None

        self.traffic_waypoint = None

        rospy.loginfo('WaypointUpdater started')

        rospy.spin()

    def publish(self, waypoints):
        l = Lane()
        l.header.frame_id = '/world'
        l.header.stamp = rospy.Time.now()
        l.waypoints = waypoints
        self.final_waypoints_pub.publish(l)

    def get_yaw(self, pose1):
        quaternion = (
            pose1.orientation.x,
            pose1.orientation.y,
            pose1.orientation.z,
            pose1.orientation.w)
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
        return yaw

    def get_direction(self, pose1, pose2):
        return math.atan2(pose2.position.y - pose1.position.y, pose2.position.x - pose1.position.x)

    def is_infront(self, wp_pose1, pose2):
        direction = self.get_direction(wp_pose1, pose2)
        return abs(direction) < math.pi*0.5 
        
    def pose_cb(self, pose):
        #rospy.loginfo("Pose callback called!!!!")
        current_pose = pose
        yaw = self.get_yaw(current_pose.pose)

        if self.lane is None:
            return

        final_waypoints = []

        start_wp, distance = self.closest_waypoint(current_pose.pose, self.lane.waypoints)

        waypoints_len = len(self.lane.waypoints)
        rospy.loginfo('Closest waypoint calculated index %s distance %s', start_wp, distance)

        for wp in range(LOOKAHEAD_WPS):
            #rospy.loginfo("Adding waypoint at index %s", (start_wp + wp) % waypoints_len)
            p = Waypoint(self.lane.waypoints[(start_wp + wp) % waypoints_len].pose, self.lane.waypoints[(start_wp + wp) % waypoints_len].twist)
            #set a speed and play a bit around...
            p.twist.twist.linear.x = self.velocity          
            final_waypoints.append(p)

        rospy.loginfo('Finally published final_waypoints')
        self.publish(final_waypoints)


    def waypoints_cb(self, lane):
        rospy.loginfo('Waypoints received %s',len(lane.waypoints))
        self.lane = lane


    def traffic_cb(self, traffic_waypoint):
        self.traffic_waypoint = traffic_waypoint
        rospy.loginfo('Traffic waypoint received:%s',traffic_waypoint)
        pass

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

    # Find closest in front waypoint given a current pose...and a list of waypoints...
    # Let's see if we can do better here as we should recalculate this stuff
    # every time...
    def closest_waypoint(self, current_pose, waypoints):
        closest_idx = -1
        min_dist = float('inf')
        for i in range(len(waypoints)):
            way_point = waypoints[i].pose.pose
            dist = self.distance_pose_to_pose(current_pose, way_point)
            if (self.is_infront(current_pose, way_point) and dist < min_dist):
                rospy.loginfo("Point %s is infront and has dist %s ", i, dist)
                min_dist = dist
                closest_idx = i
        return closest_idx, min_dist

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
