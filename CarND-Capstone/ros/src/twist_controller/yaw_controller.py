from math import atan, isnan

class YawController(object):
    def __init__(self, wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle):
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.min_speed = min_speed
        self.max_lat_accel = max_lat_accel

        self.min_angle = -max_steer_angle
        self.max_angle = max_steer_angle


    def get_angle(self, radius):
        if isnan(radius):
            angle = 0.0
        else:
            angle = atan(self.wheel_base / radius) * self.steer_ratio
        #return max(self.min_angle, min(self.max_angle, angle))
        if angle > self.max_angle:
            return self.max_angle
        elif angle < self.min_angle:
            return self.min_angle
        else:
            return angle

    def get_steering(self, cmd_vx, cmd_wz, speed):
        cmd_wz = speed * cmd_wz / cmd_vx if abs(cmd_vx) > 0. else 0.

        if abs(speed) > 0.1:
            max_yaw_rate = abs(self.max_lat_accel / speed);
            if (cmd_wz > max_yaw_rate):
                cmd_wz = max_yaw_rate
            elif cmd_wz < -max_yaw_rate:
                cmd_wz = -max_yaw_rate
            #cmd_wz = max(-max_yaw_rate, min(max_yaw_rate, cmd_wz))
        return self.get_angle(max(speed, self.min_speed) / cmd_wz) if abs(cmd_wz) > 0. else 0.0
