import cv2
import numpy as np

# Load the segmented image (the mask)
index = 3

segmented_image = cv2.imread('data/segmented_image' + str(index) + '.png', cv2.IMREAD_GRAYSCALE)

_, binary_image = cv2.threshold(segmented_image, 125, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def find_topmost_point(contour):
    topmost = tuple(contour[contour[:, :, 1].argmin()][0])
    return topmost

min_contour_length = 50 

# Filter contours by length and find the starting points
filtered_contours = [cnt for cnt in contours if cv2.arcLength(cnt, True) > min_contour_length]
starting_points = [find_topmost_point(contour) for contour in filtered_contours]

print("Filtered starting points for tracing:", starting_points)

original_image = cv2.imread('data/image' + str(index) + '.png')
original_image = cv2.resize(original_image,(512,512))

for contour in filtered_contours:
    cv2.drawContours(original_image, [contour], -1, (255, 0, 0), 2)  # Blue contours

for point in starting_points:
    cv2.circle(original_image, point, 5, (0, 255, 0), -1)  # Green starting points

cv2.imshow('Starting Points', original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
