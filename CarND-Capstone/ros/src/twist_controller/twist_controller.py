import math

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, wheel_base):
        self.wheel_base = wheel_base
        
    def control_steering(self, velocity, omega):
        if omega == 0 or velocity == 0:
            return 0
        radius = velocity / omega
        
        return math.atan2(self.wheel_base , radius)
