from isaacsim.sensors.camera import Camera
import cv2 as cv

cam_L = Camera("/World/agent/perception_sensor_mounting/camera_sensor/RSD455/Camera_OmniVision_OV9782_Left", frequency=10, resolution=(1280, 720))
cam_L.initialize()
cv.imwrite("/home/ondra/Downloads/pokus.png", cam_L.get_rgb()[:,:,[2,1,0]])

