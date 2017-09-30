#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped, PoseStamped
from styx_msgs.msg import Lane
import math
import numpy as np
import tf
import copy

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''
UPDATE_RATE = 10.0 # 10Hz
TIMEOUT_VALUE = 10.0

class DBWNode(object):
    def __init__(self):
        rospy.loginfo('DBWNode::__init__ - Start')
        rospy.init_node('dbw_node')

        rospy.loginfo('DBWNode::__init__ - Reading parameters...')
        self.vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        self.fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        self.brake_deadband = rospy.get_param('~brake_deadband', .1)
        self.decel_limit = rospy.get_param('~decel_limit', -5)
        self.accel_limit = rospy.get_param('~accel_limit', 1.)
        self.wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        self.wheel_base = rospy.get_param('~wheel_base', 2.8498)
        self.steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        self.max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        self.max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)

        #subscribe to all the topics you need to
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dwb_enable_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/final_waypoints', Lane, self.waypoints_cb)

        self.cte_counter = 0
        self.tot_cte = 0

        #create `TwistController` object
        self.controller = Controller(self.vehicle_mass, self.fuel_capacity, self.brake_deadband, self.decel_limit, self.accel_limit, \
                                        self.wheel_radius, self.wheel_base, self.steer_ratio, self.max_lat_accel, self.max_steer_angle)

        self.twist_cmd = None
        self.current_velocity = None
        self.dbw_enable_status = True
        self.time_last_cmd = None
        self.pose = None
        self.waypoints = None

        self.tf_listener = tf.TransformListener()

        rospy.loginfo('DBWNode::__init__ - entering loop')
        self.loop()

    def twist_cb(self, msg):
        self.time_last_cmd = rospy.rostime.get_time()
        self.twist_cmd = msg

    def dwb_enable_cb(self, msg):
        rospy.loginfo('DBWNode::dwb_enable_cb - received DBW enable message: %s', msg.data)
        if msg != None:
            self.dbw_enable_status = msg.data
        
    def current_velocity_cb(self, msg):
        if self.current_velocity != None:
            raw_accel = UPDATE_RATE * (msg.twist.linear.x - self.current_velocity.twist.linear.x)
            self.controller.filter_accel_value(raw_accel)
        self.current_velocity = msg

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, msg):
        self.waypoints = msg
        
    def loop(self):
        rate = rospy.Rate(UPDATE_RATE)
        rospy.loginfo("start loop %s", UPDATE_RATE)
        while not rospy.is_shutdown():
            current_time = rospy.rostime.get_time()
            rospy.loginfo("loop %s",current_time)
            if self.time_last_cmd is None:
                self.time_last_cmd = current_time
                continue

            # no need to test time_last_cmd since it is assigned together with twist_cmd
            if self.waypoints != None and self.twist_cmd != None and self.current_velocity != None:
                
                # Create lists of x and y values of the next waypoints to fit a polynomial
                x = []
                y = []
                i = 0
                # Due to race conditions, we need to store the waypoints temporary
                temp_waypoints = copy.deepcopy(self.waypoints)
                while len(x) < 20 and i < len(temp_waypoints.waypoints):
                    # Transform waypoint to car coordinates
                    temp_waypoints.waypoints[i].pose.header.frame_id = temp_waypoints.header.frame_id
                    self.tf_listener.waitForTransform("/base_link", "/world", rospy.Time(0), rospy.Duration(TIMEOUT_VALUE))
                    transformed_waypoint = self.tf_listener.transformPose("/base_link", temp_waypoints.waypoints[i].pose)
                    # Just add the x coordinate if the car did not pass the waypoint yet
                    if transformed_waypoint.pose.position.x >= 0.0:
                        x.append(transformed_waypoint.pose.position.x)
                        y.append(transformed_waypoint.pose.position.y)
                    i += 1
                coefficients = np.polyfit(x, y, 3)
                # We have to calculate the cte for a position ahead, due to delay
                cte = np.polyval(coefficients, 0.7 * self.current_velocity.twist.linear.x)
                cte *= abs(cte)
                rospy.loginfo('cte: %s', cte)
                self.tot_cte += abs(cte)
                self.cte_counter += 1
                rospy.loginfo('avg_cte: %s', self.tot_cte / self.cte_counter)
                throttle_val, brake_val, steering_val = self.controller.control(current_time, self.time_last_cmd, float(1.0/UPDATE_RATE), self.twist_cmd, \
                                                        self.current_velocity, self.dbw_enable_status, self.brake_deadband, cte)
                if self.dbw_enable_status == True:
                    self.publish(throttle_val, brake_val, steering_val)
                else:
                    rospy.loginfo("dbw_enable_status %s", self.dbw_enable_status)

            #if dbw is enabled:
            rate.sleep()

    def publish(self, throttle, brake, steer):
        rospy.loginfo('DBWNode::publish - publishing throttle (%s), brake (%s) and steer (%s) values twist_cmd x %s z %s', \
                        throttle, brake, steer,self.twist_cmd.twist.linear.x, self.twist_cmd.twist.angular.z)
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        #rospy.loginfo('DBWNode::publish - publishing throttle command')
        if brake == 0.0 and throttle > 0.0: 
            self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        #rospy.loginfo('DBWNode::publish - publishing steering command')
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.boo_cmd = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        #rospy.loginfo('DBWNode::publish - publishing brake command')
        if brake != 0.0:
            self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()

