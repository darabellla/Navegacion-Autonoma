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

#Lines processing
def average_slope_intercept(lines):
    left_lines = [] #(slope, intercept)
    left_weights = [] #(length,)
    right_lines = [] #(slope, intercept)
    right_weights = [] #(length,)

    if lines is None:
        return (None, None), (None, None)

    for line in lines:
        for x1, y1, x2, y2 in line:
            # if x1 == x2:
            # 	continue
            # calculating slope of a line
            slope = (y2 - y1) / ((x2 - x1) if x1 != x2 else 5)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + (((x2 - x1) if x1 != x2 else 5) ** 2))
            # slope of left lane is negative and for right lane slope is positive
            if (slope < 0) and (x1 < 128) and ((x2 < 128)):
                left_lines.append((slope, intercept))
                left_weights.append((length))
            elif (slope > 0) and (x1 > 128) and ((x2 > 128)):
                right_lines.append((slope, intercept))
                right_weights.append((length))
    # 
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else (None, None)
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else (None, None)
    return left_lane, right_lane

def pixel_points(y1, y2, line):
	if line is None:
		return None
	slope, intercept = line
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	y1 = int(y1)
	y2 = int(y2)
	return x1, y1, x2, y2

def lane_lines(lines, image_high):
    left_lane, right_lane = average_slope_intercept(lines)
    left_slope, right_slope = left_lane[0], right_lane[0]
    left_slope = left_slope if left_slope is None else None if left_slope == 0 else left_slope 
    right_slope = right_slope if right_slope is None else None if right_slope == 0 else right_slope 
    
    y1 = image_high
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane) if left_slope is not None else None
    right_line = pixel_points(y1, y2, right_lane) if right_slope is not None else None
    return (left_line, right_line), (left_slope, right_slope)

	
def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=12):
	line_image = np.zeros_like(image)
	for line in lines:
		if line is not None:
			cv2.line(line_image, *line, color, thickness)
	return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)

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
    # vertices = np.array([[(0,64),(48,42),(80,42),(128,64)]])
    vertices = np.array([[(0,128),(96,84),(160,84),(256,128)]])
    # vertices = np.array([[(0,128),(0,84),(256,84),(256,128)]])
    # vertices = np.array([[(0,128),(0,0),(256,0),(256,128)]])
    img_roi = np.zeros_like(img_edge)
    cv2.fillPoly(img_roi, vertices, 255)
    img_mask = cv2.bitwise_and(img_edge, img_roi)
    return img_mask

def line_cv2(img_mask):
    rho = 1             # resolución de rho en pixeles
    theta = np.pi/180   # resolución de theta en radianes
    threshold = 15     # mínimo número de votos para ser considerado una línea
    min_line_len = 15   # mínimo número de pixeles para que se forme una línea
    max_line_gap = 20   # mínimo espacio en pixeles entre segmentos de línea
    lines = cv2.HoughLinesP(img_mask, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    # se crea un fondo negro del tamaño de la imagen con bordes
    img_lines = np.zeros((img_mask.shape[0], img_mask.shape[1], 3), dtype=np.uint8)

    both_lane_lines, both_lane_slopes = lane_lines(lines, img_lines.shape[0])
    print(both_lane_lines)
    print(both_lane_slopes)
    # se dibuja cada una de las líneas sobre la imagen con fondo negro
    for line in both_lane_lines:
        if line is None:
             continue
        x1,y1,x2,y2 = line
        # for x1,y1,x2,y2 in line:
        cv2.line(img_lines, (x1,y1), (x2,y2), [0, 255, 0], 3)
    return img_lines, both_lane_slopes

def lane_cv2(image):
    rgb_image = rgb_cv2(image)
    grey_image = greyscale_cv2(image)
    edge_image = canny_edges_cv2(grey_image)
    mask_image = mask_cv2(edge_image)
    line_image, both_lane_slopes = line_cv2(mask_image)

    auto_steering_angle = calc_auto_steering_angle(both_lane_slopes[0], both_lane_slopes[1])

    alpha = 1
    beta = 1
    gamma = 1
    img_lane_lines = cv2.addWeighted(rgb_image, alpha, line_image, beta, gamma)

    return img_lane_lines, auto_steering_angle

def calc_auto_steering_angle(left_lane_slope, right_lane_slope):
    global auto_steering

    left_steering = np.arctan(left_lane_slope) if left_lane_slope is not None else None
    right_steering = np.arctan(right_lane_slope) if right_lane_slope is not None else None

    auto_steering_angle = (left_steering + np.pi/2 if left_steering is not None else 0)
    auto_steering_angle += (right_steering - np.pi/2 if right_steering is not None else 0)

    if auto_steering == np.pi:
         auto_steering = auto_steering_angle
         inc_steering_angle = 0
    else:
        if (left_steering is not None) and (right_steering is not None):
            inc_steering_angle = auto_steering_angle - auto_steering
            auto_steering = auto_steering_angle
        else:
            auto_steering = 0
            inc_steering_angle = auto_steering_angle

    # auto_steering_angle = auto_steering_angle + np.pi if auto_steering_angle < 0 else auto_steering_angle

    # auto_steering_angle = np.pi/2 - auto_steering_angle 

    # auto_steering_angle = auto_steering_angle if auto_steering_angle != np.pi/2 else 0

    return  inc_steering_angle

#Display image 
def display_image(display, image, isgray=True):
    # Image to display
    if isgray:
        image_rgb = np.dstack((image, image,image,))
    else:
        image_rgb = image
    # Display image
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

#initial angle and speed 
auto_steering = np.pi
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
        lane_image, steer_angle = lane_cv2(image)
        display_image(display_img, lane_image, False)
        print(lane_image.shape)
        if steer_angle is not None:
             change_steer_angle(steer_angle)
             print("AUTO right" if steer_angle > 0 else "Auto left")
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