import cv2
import numpy as np

# Paths to the model files (ensure these files are downloaded and placed in the 'models' folder)
prototxt = 'models/colorization_deploy_v2.prototxt'  # Path to the Caffe prototxt file which defines the model architecture
model = 'models/colorization_release_v2.caffemodel'  # Path to the pre-trained model weights file
points = 'models/pts_in_hull.npy'  # Path to the numpy file containing the cluster centers for colorization

# Load the Caffe model using OpenCV's DNN module
net = cv2.dnn.readNetFromCaffe(prototxt, model)
pts = np.load(points)  # Load the cluster centers from the numpy file

# Add the cluster centers as 1x1 convolutions to the model
class8 = net.getLayerId("class8_ab")  # Get the layer ID for the "class8_ab" layer
conv8 = net.getLayerId("conv8_313_rh")  # Get the layer ID for the "conv8_313_rh" layer
pts = pts.transpose().reshape(2, 313, 1, 1)  # Reshape the points to match the required input dimensions
net.getLayer(class8).blobs = [pts.astype("float32")]  # Assign the points as blobs to the "class8_ab" layer
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]  # Assign a constant blob to the "conv8_313_rh" layer

# Load the input image (grayscale or black-and-white)
input_image_path = 'input.jpg'  # Path to the input black-and-white image
output_image_path = 'output.jpg'  # Path where the colorized image will be saved
image = cv2.imread(input_image_path)  # Read the input image using OpenCV

# Convert the image to Lab color space
scaled = image.astype("float32") / 255.0  # Scale the pixel values to the range [0, 1]
lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)  # Convert the image from BGR to Lab color space

# Resize the Lab image to 224x224 (the dimensions the model was trained on)
resized = cv2.resize(lab, (224, 224))  # Resize the Lab image to 224x224 pixels
L = cv2.split(resized)[0]  # Extract the L channel (lightness) from the resized Lab image
L -= 50  # Subtract 50 to normalize the L channel as required by the model

# Pass the L channel through the network to predict the ab channels
net.setInput(cv2.dnn.blobFromImage(L))  # Set the L channel as the input to the network
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))  # Run forward pass to predict the ab channels

# Resize the predicted ab component to match the original image size
ab = cv2.resize(ab, (image.shape[1], image.shape[0]))  # Resize the predicted ab channels back to the original image size

# Combine the original L component with the predicted ab components
L = cv2.split(lab)[0]  # Extract the original L channel from the Lab image
colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)  # Concatenate the L channel with the predicted ab channels

# Convert the colorized image from Lab to BGR
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)  # Convert the image back from Lab to BGR color space
colorized = np.clip(colorized, 0, 1)  # Clip the pixel values to ensure they are within the valid range [0, 1]

# Convert the image back to the 0-255 range
colorized = (255 * colorized).astype("uint8")  # Scale the pixel values back to the range [0, 255] and convert to uint8

# Save the colorized image
cv2.imwrite(output_image_path, colorized)  # Save the colorized image to the specified path

print(f"Colorized image saved at {output_image_path}")  # Print a message indicating where the image has been saved
