import PIL
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications import resnet50

filename = './images/banana.jfif'

#load an image in PIL format
original = load_img(filename, target_size = (224, 224))
print('PIL image size', original.size)

# Convert the Pil image to a numpy array

numpy_image = img_to_array(original)
plt.imshow(np.uint8(numpy_image))
plt.show()

print('numpy array size', numpy_image.shape)

#Convet the image/images into batch format
image_batch = np.expand_dims(numpy_image, axis = 0)

print('image batch size', image_batch.shape)

processed_image = resnet50.preprocess_input(image_batch.copy())

# Create resnet model
resnet_model = resnet50.ResNet50(weights = 'imagenet')

predictions = resnet_model.predict(processed_image)

label = decode_predictions(predictions)

print(label)       