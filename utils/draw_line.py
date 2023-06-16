import cv2
import numpy as np

# Function to handle mouse events
def handle_click(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))

# Ask user for number of lines
num_lines = int(input("How many lines do you want to draw? "))


# Read the picture
image_path = input("Enter the path to the image file: ")
image = cv2.imread(image_path)

# Create a window and set mouse callback
cv2.namedWindow('image')
cv2.setMouseCallback('image', handle_click)

points = []

while True:
    # Show image
    cv2.imshow('image', image)
    key = cv2.waitKey(1) & 0xFF

    # Break loop if user presses 'q'
    if key == ord('q'):
        break

    # Check if enough points have been collected
    if len(points) == num_lines * 2:
        break

# Print coordinates of clicked points
for i in range(num_lines):
    print(f"Line {i+1} start x: {points[2*i][0]}")
    print(f"Line {i+1} start y: {points[2*i][1]}")
    print(f"Line {i+1} end x: {points[2*i+1][0]}")
    print(f"Line {i+1} end y: {points[2*i+1][1]}")

cv2.destroyAllWindows()
