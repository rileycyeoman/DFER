import cv2
import dlib

import torchvision.datasets as datasets

# Path to the folder containing the images
data_folder = 'content/dataset/train/'

# Create an instance of ImageFolder dataset
dataset = datasets.ImageFolder(root=data_folder)

# Load the pre-trained facial landmark predictor
predictor = dlib.shape_predictor("path/to/shape_predictor_68_face_landmarks.dat")

# Iterate through each image in the dataset
for image_path, _ in dataset:
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale image
    faces = detector(gray)
    
    # Iterate through each detected face
    for face in faces:
        # Predict the facial landmarks
        landmarks = predictor(gray, face)
        
        # Create a blank heatmap image
        heatmap = np.zeros_like(gray)
        
        # Iterate through each landmark point
        for point in landmarks.parts():
            # Get the x and y coordinates of the landmark point
            x, y = point.x, point.y
            
            # Draw a circle at the landmark point on the heatmap
            cv2.circle(heatmap, (x, y), 1, (255, 255, 255), -1)
        
        # Display the heatmap
        cv2.imshow("Facial Landmark Heatmap", heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
