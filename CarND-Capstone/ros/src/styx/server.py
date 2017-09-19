#!/usr/bin/env python

import socketio
import eventlet
import eventlet.wsgi
import time
from flask import Flask, render_template

from bridge import Bridge
from conf import conf
import rospy

sio = socketio.Server()
app = Flask(__name__)
msgs = {}

dbw_enable = False

@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)

def send(topic, data):
    s = 1
    rospy.loginfo("send %s",topic)
    msgs[topic] = data
    #sio.emit(topic, data=json.dumps(data), skip_sid=True)

bridge = Bridge(conf, send)

@sio.on('telemetry')
def telemetry(sid, data):
    global dbw_enable
    if data["dbw_enable"] != dbw_enable:
        dbw_enable = data["dbw_enable"]
        bridge.publish_dbw_status(dbw_enable)
        rospy.loginfo("publish dbw_enable %s",dbw_enable)
    bridge.publish_odometry(data)
    if len(msgs) == 0:
      return
    rospy.loginfo("telemetry %s messages",len(msgs))
    for i in range(len(msgs)):
        topic, data = msgs.popitem()
        sio.emit(topic, data=data, skip_sid=True)
  
@sio.on('control')
def control(sid, data):
    bridge.publish_controls(data)

@sio.on('obstacle')
def obstacle(sid, data):
#    bridge.publish_obstacles(data)
#    rospy.loginfo("obstacle")
    pass
    

@sio.on('lidar')
def obstacle(sid, data):
#    rospy.loginfo("lidar")
#    bridge.publish_lidar(data)
    pass

@sio.on('trafficlights')
def trafficlights(sid, data):
#    rospy.loginfo("trafficlights")
    bridge.publish_traffic(data)
#    pass

@sio.on('image')
def image(sid, data):
#    rospy.loginfo("image")
    bridge.publish_camera(data)
#    pass

if __name__ == '__main__':

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
