"""camera_pid controller."""
from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import csv
import json
from keras.models import model_from_json


# Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

# Image processing
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

# Display image
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image, image))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

# Initial angle and speed
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 30

# Set target speed
def set_speed(kmh):
    global speed
    speed = kmh

# Update steering angle
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle

    # Limit range of the steering angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # Update steering angle
    angle = wheel_angle

# Validate increment of steering angle
def change_steer_angle(inc):
    global manual_steering
    # Apply increment
    new_manual_steering = manual_steering + inc
    # Validate interval
    if new_manual_steering <= 25.0 and new_manual_steering >= -25.0:
        manual_steering = new_manual_steering
        set_steering_angle(manual_steering * 0.02)
    # Debugging
    if manual_steering == 0:
        print("going straight")
    else:
        turn = "left" if steering_angle < 0 else "right"
        print("turning {} rad {}".format(str(steering_angle), turn))
 
#   preprocesamiento de la imagen  -------------------------------------   
def img_preprocess(image):
    img = image[30:58,:]
    # Convert grayscale image to RGB by duplicating the channels
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Convert RGB to YUV
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img, (3,3), 0)
    img = cv2.resize(img, (128, 32))
    img = img/255
    return img



# Main function
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)

    # Processing display
    display_img = Display("display")

     #create keyboard instance---------------
    keyboard=Keyboard()
    keyboard.enable(timestep)

    image_counter = 0
    capture_interval = 100  # Adjust this value as needed for the interval in milliseconds
    
    #-----------------------------lidar----------------------------
    lidar=robot.getLidar('lidar')
    lidar.enable(timestep)
    lidar.enablePointCloud()
    
    
    #cargar el json tiene la estructura de la red neuronal-----------------------------------
    json_file = open("sdcg11_model_v1.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    model_v11 = model_from_json(loaded_model_json)
    model_v11.load_weights('sdcg11_model_weights_v1.h5')
    #----------------------------------------------------------------------------------------

    while robot.step() != -1:
        
        #----------------------------------lidar
        range_image=lidar.getRangeImage()

        # Get image from camera
        image = get_image(camera)

        # Process and display image
        grey_image = greyscale_cv2(image)
        display_image(display_img, grey_image)

        key=keyboard.getKey()

        if key == keyboard.UP: #up
            set_speed(speed + 5.0)
        elif key == keyboard.DOWN: #down
            set_speed(speed - 5.0)
        elif key == keyboard.RIGHT: #right
            change_steer_angle(+1)
        elif key == keyboard.LEFT: #left
            change_steer_angle(-1)

        # Save image and steering angle at regular intervals

        model_v11.predict(np.asarray([img_preprocess(grey_image)]))
        
        steringAngle = model_v11[0][0]
        # Update angle and speed
        driver.setSteeringAngle(steringAngle)
        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()
