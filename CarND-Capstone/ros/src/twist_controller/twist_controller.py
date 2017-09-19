
import math
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
WEIGHT_PERSON = 75
MIN_SPEED = 2.0 * ONE_MPH

#STEER_Kp = 0.0
STEER_Kp = 0.0
STEER_Ki = 0.0
STEER_Kd = 0.0

VELOCITY_Kp = 2.0
VELOCITY_Ki = 0.0
VELOCITY_Kd = 0.0

ACCEL_Kp = 0.4
ACCEL_Ki = 0.1
ACCEL_Kd = 0.0

ACCEL_Tau = 0.5
ACCEL_Ts = 0.02

class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        self.vehicle_mass = args[0]
        self.fuel_capacity = args[1]
        self.brake_deadband = args[2]
        self.decel_limit = args[3]
        self.accel_limit = args[4]
        self.wheel_radius = args[5]
        self.wheel_base = args[6]
        self.steer_ratio = args[7]
        self.max_lat_accel = args[8]
        self.max_steer_angle = args[9]

        self.velocity_pid = PID(VELOCITY_Kp, VELOCITY_Ki, VELOCITY_Kd, -abs(self.decel_limit), abs(self.accel_limit))
        self.accel_pid = PID(ACCEL_Kp, ACCEL_Ki, ACCEL_Kd)
        self.steering_cntrl = YawController(self.wheel_base, self.steer_ratio, MIN_SPEED, self.max_lat_accel, self.max_steer_angle)
        self.steering_pid = PID(STEER_Kp, STEER_Ki, STEER_Kd)

        self.accel_lpf = LowPassFilter(ACCEL_Tau, ACCEL_Ts)
        #pass

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        #return 1., 0., 0.
        current_time = args[0]
        last_cmd_time = args[1]
        control_period = args[2]
        twist_cmd = args[3]
        current_velocity = args[4]
        dbw_enabled = args[5]

        if (current_time - last_cmd_time) > 10 * control_period:
            self.velocity_pid.reset()
            self.accel_pid.reset()
            self.steering_pid.reset()

        vehicle_mass = self.vehicle_mass + self.fuel_capacity * GAS_DENSITY + 2 * WEIGHT_PERSON

        velocity_error = twist_cmd.twist.linear.x - current_velocity.twist.linear.x

        if abs(twist_cmd.twist.linear.x) < (1.0 * ONE_MPH):
            self.velocity_pid.reset()

        accel_cmd = self.velocity_pid.step(velocity_error, control_period)

        if twist_cmd.twist.linear.x <= 1e-2:
            accel_cmd = min(accel_cmd, -530 / vehicle_mass / self.wheel_radius)
        elif twist_cmd.twist.linear.x < MIN_SPEED:
            twist_cmd.twist.angular.z = twist_cmd.twist.angular.z * MIN_SPEED / twist_cmd.twist.linear.x
            twist_cmd.twist.linear.x = MIN_SPEED

        if dbw_enabled:
            if accel_cmd >= 0:
                throttle_val = self.accel_pid.step(accel_cmd - self.accel_lpf.get(), control_period)
            else:
                throttle_val = 0.0
                self.accel_pid.reset()

            if accel_cmd < -self.brake_deadband or twist_cmd.twist.linear.x < MIN_SPEED:
                brake_val = -accel_cmd * vehicle_mass * self.wheel_radius
            else:
                brake_val = 0.0

            steering_val = self.steering_cntrl.get_steering(twist_cmd.twist.linear.x, twist_cmd.twist.angular.z, current_velocity.twist.linear.x) + self.steering_pid.step(twist_cmd.twist.angular.z - current_velocity.twist.angular.z, control_period)

            steering_val = max(-self.max_steer_angle, min(self.max_steer_angle, steering_val))

            return throttle_val, brake_val, steering_val
        else:
            self.velocity_pid.reset()
            self.accel_pid.reset()
            self.steering_pid.reset()
            return 0.0, 0.0, 0.0

    def filter_accel_value(self, value):
        self.accel_lpf.filt(value)

    def get_filtered_accel(self):
        return self.accel_lpf.get()
