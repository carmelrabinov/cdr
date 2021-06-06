import cv2
import numpy as np


def get_cube_mask(image, use_BGR: bool = False):
    color_trasformation = cv2.COLOR_BGR2HSV if use_BGR else cv2.COLOR_RGB2HSV
    hsv = cv2.cvtColor(image, color_trasformation)
    # view_image(image)  ## 1

    # Range for lower red
    lower_red = np.array([0, 100, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # Range for upper range
    lower_red = np.array([170, 100, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Generating the final mask to detect red color
    mask = mask1 + mask2

    # Generate intermediate image; use morphological closing to keep parts of the brain together
    inter = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find largest contour in intermediate image
    cnts, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)

    # Output clean_mask
    clean_mask = np.zeros(image.shape[:2], np.uint8)
    cv2.drawContours(clean_mask, [cnt], -1, 255, cv2.FILLED)
    # clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

    # Segmenting the cloth out of the frame using bitwise and with the inverted mask
    # res1 = cv2.bitwise_and(image, image, mask=clean_mask)
    # view_image(image*np.expand_dims(clean_mask, axis=2))  ## 1

    return mask, clean_mask


def get_cube_pixel_coordinates(image, use_BGR: bool = False):
    mask, clean_mask = get_cube_mask(image, use_BGR=use_BGR)
    assert clean_mask.any(), "cube was not detected"
    x_ind, y_ind = np.where(clean_mask)
    x = x_ind.mean()
    y = y_ind.mean()

    return [x, y]


def extract_point_pixel_coordinates(image, use_BGR: bool = False):
    """extracts 4 red pixel points """
    mask, clean_mask = get_cube_mask(np.array(image), use_BGR=use_BGR)
    x_ind, y_ind = np.where(mask)

    points = [[], [], [], []]
    point_cnt = 0
    recognized_points = 0
    while point_cnt < len(x_ind):
        next_point = np.array([x_ind[point_cnt], y_ind[point_cnt]])

        # add the first point
        if recognized_points == 0:
            points[recognized_points].append(next_point)
            recognized_points += 1
        else:
            dist_from_prev_points = [np.linalg.norm(next_point - np.mean(points[i], axis=0)) for i in range(recognized_points)]
            point_clusters = np.array(dist_from_prev_points) < 6
            # add pixel to previous point if is it close to it
            if np.any(point_clusters):
                cluster = np.where(point_clusters==True)[0][0]
                points[cluster].append(next_point)
            # add pixel to a new point
            elif recognized_points < 4:
                points[recognized_points].append(next_point)
                recognized_points += 1
            else:
                assert False, "Error, there should only be 4 point to recognize"
        point_cnt += 1


    pixel_coordinates = [np.mean(p, axis=0) for p in points]
    return pixel_coordinates


def generate_calibration_coordinates(image, ordered_pixel_coord=None):
    if ordered_pixel_coord is None:
        pixel_coord = extract_point_pixel_coordinates(image)
        ordered_pixel_coord = np.float32(sorted(pixel_coord, key=lambda x: x[0], reverse=False))
    world_coord = np.float32([[0.35, 0.1], [0.45, 0.2], [0.5, -0.15], [0.6, -0.05]])
    M = cv2.getPerspectiveTransform(ordered_pixel_coord, world_coord)
    return M


def generate_oposite_calibration_coordinates(image, ordered_pixel_coord=None):
    if ordered_pixel_coord is None:
        pixel_coord = extract_point_pixel_coordinates(image)
        ordered_pixel_coord = np.float32(sorted(pixel_coord, key=lambda x: x[0], reverse=False))
    world_coord = np.float32([[0.35, 0.1], [0.45, 0.2], [0.5, -0.15], [0.6, -0.05]])
    M = cv2.getPerspectiveTransform(world_coord, ordered_pixel_coord)
    return M

