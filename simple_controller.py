"""camera_pid controller."""

from controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os

#Getting image from camera
def get_image(camera):
    raw_image = camera.getImage()  
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image

#Image processing
def greyscale_cv2(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_img

def rgb_cv2(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return rgb_img

def canny_edges_cv2(img_gray):
    img_blur = cv2.GaussianBlur(img_gray, (3,3), 0, 0)
    img_canny = cv2.Canny(img_blur, 40, 120)
    return img_canny

def mask_cv2(img_edge):
    # vertices = np.array([[(10,64),(42,30),(86,30),(118,64)]])
    vertices = np.array([[(0,64),(48,45),(80,45),(128,64)]])
    img_roi = np.zeros_like(img_edge)
    cv2.fillPoly(img_roi, vertices, 255)
    img_mask = cv2.bitwise_and(img_edge, img_roi)
    return img_mask

def line_cv2(img_mask):
    rho = 1             # resolución de rho en pixeles
    theta = np.pi/180   # resolución de theta en radianes
    threshold = 5     # mínimo número de votos para ser considerado una línea
    min_line_len = 1   # mínimo número de pixeles para que se forme una línea
    max_line_gap = 0   # mínimo espacio en pixeles entre segmentos de línea
    lines = cv2.HoughLinesP(img_mask, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # se crea un fondo negro del tamaño de la imagen con bordes
    img_lines = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)
    # se dibuja cada una de las líneas sobre la imagen con fondo negro
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img_lines, (x1,y1), (x2,y2), [0, 255, 0], 30)
    return img_lines

def lane_cv2(image):
    rgb_image = rgb_cv2(image)
    grey_image = greyscale_cv2(image)
    edge_image = canny_edges_cv2(grey_image)
    mask_image = mask_cv2(edge_image)
    line_image = line_cv2(mask_image)

    alpha = 1
    beta = 1
    gamma = 1
    img_lane_lines = cv2.addWeighted(rgb_image, alpha, line_image, beta, gamma)

    return mask_image     

#Display image 
def display_image(display, image):
    # Image to display
    image_rgb = np.dstack((image, image,image,))
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#initial angle and speed 
manual_steering = 0
steering_angle = 0
angle = 0.0
speed = 30

# set target speed
def set_speed(kmh):
    global speed            #robot.step(50)
#update steering angle
def set_steering_angle(wheel_angle):
    global angle, steering_angle
    # Check limits of steering
    if (wheel_angle - steering_angle) > 0.1:
        wheel_angle = steering_angle + 0.1
    if (wheel_angle - steering_angle) < -0.1:
        wheel_angle = steering_angle - 0.1
    steering_angle = wheel_angle
  
    # limit range of the steering angle
    if wheel_angle > 0.5:
        wheel_angle = 0.5
    elif wheel_angle < -0.5:
        wheel_angle = -0.5
    # update steering angle
    angle = wheel_angle

#validate increment of steering angle
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
        print("turning {} rad {}".format(str(steering_angle),turn))

# main
def main():
    # Create the Robot instance.
    robot = Car()
    driver = Driver()

    # Get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # Create camera instance
    camera = robot.getDevice("camera")
    camera.enable(timestep)  # timestep

    # processing display
    display_img = Display("display_image")

    #create keyboard instance
    keyboard=Keyboard()
    keyboard.enable(timestep)

    while robot.step() != -1:
        # Get image from camera
        image = get_image(camera)

        # Process and display image 
        lane_image = lane_cv2(image)
        display_image(display_img, lane_image)
        print(lane_image.shape)
        # Read keyboard
        key=keyboard.getKey()
        if key == keyboard.UP: #up
            set_speed(speed + 5.0)
            print("up")
        elif key == keyboard.DOWN: #down
            set_speed(speed - 5.0)
            print("down")
        elif key == keyboard.RIGHT: #right
            change_steer_angle(+1)
            print("right")
        elif key == keyboard.LEFT: #left
            change_steer_angle(-1)
            print("left")
        elif key == ord('A'):
            #filename with timestamp and saved in current directory
            current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
            file_name = current_datetime + ".png"
            print("Image taken")
            camera.saveImage(os.getcwd() + "/" + file_name, 1)
            
        #update angle and speed
        driver.setSteeringAngle(angle)
        driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()