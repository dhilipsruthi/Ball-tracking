# Billiard balls tracking on a pool table
# By
# Dhilip kumar

import cv2
import numpy as np

def main():

    # Open the video file
    cap = cv2.VideoCapture("Input.mp4")

    # Initialize Variable

    # Get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Variables used to execute certain part of code only once
    first_read = True
    first_ball = True

    white_ball_arr = []
    red_ball_arr = []
    green_ball_arr = []
    black_ball_arr = []
    yellow_ball_arr = []

    # Dictionary that contain the HSV parameters of each ball we are dealing with
    ball_dic = {'White': (25, 90, 240), 'Black': (20, 65, 71), 'Green': (77, 209, 125), 'Yellow': (26, 220, 245),
                'Red': (1, 124, 168)}
    track_ball ={}

    # HSV limit of the background color
    background_lower = (95, 30, 80)
    background_upper = (100, 250, 180)

    # Initialize Output File
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30, (frame_width, frame_height))

    while cap.isOpened():
        # Get Frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # The below logic is executed only once
        # The logic is to read the AruCon markers and identify the x,y points that will be used in the warp perspective
        if first_read:
        # We are using 4x4 AruCon markers so we are getting their values
            arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
        # Detect the markers and store them corners and ids
            corners, ids, _ = cv2.aruco.detectMarkers(
            image=frame,
            dictionary=arucoDict
            )
        # We are making sure we have all 4 AruCon markers read
            if len(corners) < 4:
                continue
            else:
                first_read = False
        # We are drawing the ids on the markers, this is for our reference and will not be seen in the output
                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(
                        image=frame, corners=corners, ids=ids, borderColor=(0, 0, 255))

        # loop over the detected ArUCo corners and get all their x,y coordinates
                for (markerCorner, markerID) in zip(corners, ids):
                    corners = markerCorner.reshape((4, 2))
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    topRight = (int(topRight[0] + 100), int(topRight[1]+100))
                    bottomRight = (int(bottomRight[0]-100), int(bottomRight[1]-100))
                    bottomLeft = (int(bottomLeft[0]-100), int(bottomLeft[1]-100))
                    topLeft = (int(topLeft[0]+100), int(topLeft[1]+100))

        # We dont want to see the AruCon markers in the output, so we are taking points at their edges
                    if markerID[0] == 0:
                        frame_topleft = bottomRight
                    if markerID[0] == 3:
                        frame_topright = bottomLeft
                    if markerID[0] == 1:
                        frame_botleft = topRight
                    if markerID[0] == 2:
                        frame_botright = topLeft

        # Reference points of the original frame
                source_points = np.float32([frame_topleft, frame_topright, frame_botleft, frame_botright])

        # Reference points of the target frame
                target_points = np.float32([[0, 0], [frame_width, 0], [0, frame_height], [frame_width, frame_height]])

        # Find the homography
        H, _ = cv2.findHomography(source_points, target_points)

        # Perform the transformation to make the frame perpendicular to the viewer
        transformed_frame = cv2.warpPerspective(frame, H, (frame_width, frame_height))

        # Apply Gaussian Blur on the transformed frame
        transformed_blur_frame = cv2.GaussianBlur(transformed_frame, (0, 0), 2)

        # Get the HSV version of the frame
        hsv = cv2.cvtColor(transformed_blur_frame, cv2.COLOR_BGR2HSV)

        # Using the lower and upper HSV values of the table apply mask on the frame
        mask = cv2.inRange(hsv, background_lower, background_upper)

        # Perform opening and closing to
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        mask_closing = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Thresholding of the frame
        thresh, binary_img = cv2.threshold(
            mask_closing, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV)
        binary_img = binary_img.astype(np.uint8)

        # Find the contours in the frame
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Create a copy of the transformed image that can be used in future processing on average color
        copy_img = transformed_frame.copy()
        # Create a small filter for finding the average color of each ball
        K = np.ones((5, 5), np.uint8)
        # convert the image to gray and create a empty mask
        gray = cv2.cvtColor(copy_img, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, np.uint8)

        # The below logic find the contours that resemble a ball shape
        for i in range(len(contours)):

        # Get the radius and center of the contours
            (x, y), radius = cv2.minEnclosingCircle(contours[i])

        # Get the area of each contours
            area = cv2.contourArea(contours[i])
            center = (int(x), int(y))
            radius = int(radius)

        # Check if the radius and area of the contours fall under certain range to determine if its a ball
           
            if (radius <200) or (radius > 500):
                continue
            if (area < 5000) or (area > 50000):
                continue

        # Reset the mask for each ball
            mask[...] = 0
        # Draws the mask on the current ball
            cv2.drawContours(mask, contours, i, 255, -1)
        # Find the average color mean of each contour and apply it at the center point with radius as 70
            ctrs_color = cv2.circle(copy_img,center,70,cv2.mean(copy_img, mask),-1)

        # Find the color at the center
            colorsB = ctrs_color[int(center[1]), int(center[0]), 0]
            colorsG = ctrs_color[int(center[1]), int(center[0]), 1]
            colorsR = ctrs_color[int(center[1]), int(center[0]), 2]

        # Get the HSV value of that point
            temp_value = np.uint8([[[colorsB, colorsG, colorsR]]])
            hsv_value = cv2.cvtColor(temp_value, cv2.COLOR_BGR2HSV)

            h = hsv_value[0][0][0]
            s = hsv_value[0][0][1]
            v = hsv_value[0][0][2]

            for color,values in ball_dic.items():

        # Find the root mean square distance of the center points hsv value from the predefined hsv values
                h_diff = (values[0] - h) ** 2
                s_diff = (values[1] - s) ** 2
                v_diff = (values[2] - v) ** 2
                d = (h_diff + s_diff + v_diff) ** 0.5
        # When the distance is less than 45 we have identified the ball of that color
                if d < 45:
        # Write the color of the ball
                    transformed_frame = cv2.putText(transformed_frame, color, center, cv2.FONT_HERSHEY_SIMPLEX,
                                              1, (255, 0, 0), 4, cv2.LINE_AA)
        # If it is the first run not the initial position
                    if first_ball:
                        track_ball[color] = center
                    else:
        # when its not first run calculate the distance between current point and previous point
                        curr_dis = ((track_ball[color][0] - center[0]) ** 2 + (track_ball[color][1] - center[1]) ** 2) ** 0.5
        # If the ball has moved more than 100 units record the new points in the concerned list
                        if curr_dis > 100:
                            if color == 'White':
                                white_ball_arr.append(center)
                            if color == 'Red':
                                red_ball_arr.append(center)
                            if color == 'Black':
                                black_ball_arr.append(center)
                            if color == 'Green':
                                green_ball_arr.append(center)
                            if color == 'Yellow':
                                yellow_ball_arr.append(center)
        # when there is more than 2 points draw line between the points
        white_len_ball = len(white_ball_arr)
        if white_len_ball >1:
            for index in range(white_len_ball-1):
        # Use white color to draw the tracing line of white ball
                cv2.line(transformed_frame, white_ball_arr[index], white_ball_arr[index+1], (171, 233, 241), 5)
        # When there is more than 2 points draw line between the points
        yellow_len_ball = len(yellow_ball_arr)
        if yellow_len_ball > 1:
            for index in range(yellow_len_ball - 1):
        # Use yellow color to draw the tracing line of yellow ball
                cv2.line(transformed_frame, yellow_ball_arr[index], yellow_ball_arr[index + 1], (39, 207, 238), 5)
        # When there is more than 2 points draw line between the points
        black_len_ball = len(black_ball_arr)
        if black_len_ball > 1:
            for index in range(black_len_ball - 1):
        # Use black color to draw the tracing line of black ball
                cv2.line(transformed_frame, black_ball_arr[index], black_ball_arr[index + 1], (57, 80, 96), 5)
        # When there is more than 2 points draw line between the points
        red_len_ball = len(red_ball_arr)
        if red_len_ball > 1:
            for index in range(red_len_ball - 1):
        # Use red color to draw the tracing line of red ball
                cv2.line(transformed_frame, red_ball_arr[index], red_ball_arr[index + 1], (87, 90, 164), 5)
        # When there is more than 2 points draw line between the points
        green_len_ball = len(green_ball_arr)
        if green_len_ball > 1:
            for index in range(green_len_ball - 1):
        # Use green color to draw the tracing line of green ball
                cv2.line(transformed_frame, green_ball_arr[index], green_ball_arr[index + 1], (77, 209, 125), 5)

        # After first run set this to False
        first_ball = False

        # Display the frame
        cv2.imshow('Output_video', transformed_frame)

        # Write frame
        out.write(transformed_frame)

        # Wait for key
        if cv2.waitKey(1) == ord('q'):
            break
    # Release Everything
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
