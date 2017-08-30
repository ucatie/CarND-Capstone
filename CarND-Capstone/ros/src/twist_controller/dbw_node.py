#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped, PoseStamped
import math

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
UPDATE_RATE = 10.0

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
        self.max_steer_angle = math.radians(rospy.get_param('~max_steer_angle', 8.))

        rospy.loginfo('DBWNode::__init__ - creating publisher to /vehicle/steering_cmd')
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)
        rospy.loginfo('DBWNode::__init__ - creating publisher to /vehicle/throttle_cmd')
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        rospy.loginfo('DBWNode::__init__ - creating publisher to /vehicle/brake_cmd')
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)
        # TODO: Subscribe to all the topics you need to
        rospy.loginfo('DBWNode::__init__ - subscribing to /twist_cmd')
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cb)
        rospy.loginfo('DBWNode::__init__ - subscribing to /vehicle/dbw_enabled')
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dwb_enable_cb)
        rospy.loginfo('DBWNode::__init__ - subscribing to /current_velocity')
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        # rospy.loginfo('DBWNode::__init__ - subscribing to /current_pose')
        # rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb)

        # TODO: Create `TwistController` object
        # rospy.loginfo('DBWNode::__init__ - instantiating controller object')
        self.controller = Controller(self.vehicle_mass, self.fuel_capacity, self.brake_deadband, self.decel_limit, self.accel_limit, self.wheel_radius, self.wheel_base, self.steer_ratio, self.max_lat_accel, self.max_steer_angle)

        self.twist_cmd = None
        self.current_velocity = None
        self.dbw_enable_status = True
        self.time_last_cmd = None

        rospy.loginfo('DBWNode::__init__ - entering loop')
        self.loop()

    def twist_cb(self, msg):
        rospy.loginfo('DBWNode::twist_cb - received twist command -> linear: %s, angular: %s', msg.twist.linear.x, msg.twist.angular.z)
        self.time_last_cmd = rospy.rostime.get_time()
        self.twist_cmd = msg

    def dwb_enable_cb(self, msg):
        rospy.loginfo('DBWNode::dwb_enable_cb - received DBW enable message: %s', msg.data)
        if msg != None:
            self.dbw_enable_status = msg.data
        #pass

    def current_velocity_cb(self, msg):
        rospy.loginfo('DBWNode::current_velocity_cb - received current velocity message: linear %s, angular %s', msg.twist.linear.x, msg.twist.angular.z)
        if self.current_velocity != None:
            raw_accel = UPDATE_RATE * (msg.twist.linear.x - self.current_velocity.twist.linear.x)
            self.controller.filter_accel_value(raw_accel)
        self.current_velocity = msg
        #pass

    def loop(self):
        rate = rospy.Rate(UPDATE_RATE) # 50Hz
        while not rospy.is_shutdown():
            current_time = rospy.rostime.get_time()

            # no need to test time_last_cmd since it is assigned together with twist_cmd
            if self.twist_cmd != None and self.current_velocity != None:
                throttle_val, brake_val, steering_val = self.controller.control(current_time, self.time_last_cmd, float(1.0/UPDATE_RATE), self.twist_cmd, self.current_velocity, self.dbw_enable_status, self.brake_deadband)
                if self.dbw_enable_status == True:
                    self.publish(throttle_val, brake_val, steering_val)
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            # You should only publish the control commands if dbw is enabled
            # throttle, brake, steering = self.controller.control(<proposed linear velocity>,
            #                                                     <proposed angular velocity>,
            #                                                     <current linear velocity>,
            #                                                     <dbw status>,
            #                                                     <any other argument you need>)

            #if dbw is enabled:
            rate.sleep()

    def publish(self, throttle, brake, steer):
        rospy.loginfo('DBWNode::publish - publishing throttle (%s), brake (%s) and steer (%s) values', throttle, brake, steer)
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        rospy.loginfo('DBWNode::publish - publishing throttle command')
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        rospy.loginfo('DBWNode::publish - publishing steering command')
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        rospy.loginfo('DBWNode::publish - publishing brake command')
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
