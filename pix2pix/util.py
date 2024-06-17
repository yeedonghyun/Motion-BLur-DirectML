import tensorflow as tf
from matplotlib import pyplot as plt

def process_image(image, channels=3):
    image = tf.image.decode_png(image, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
  
    return image

def combine_images(paths):
    images = [process_image(tf.io.read_file(path)) for path in paths]
    combined_image = tf.concat(images, axis=-1)
    return combined_image

def parse_function(color_path, depth_path, motion_vector_path, motion_blur_path):
    paths = [color_path, depth_path, motion_vector_path, motion_blur_path]
    combined_image = combine_images(paths)
    return combined_image

def display_images(batch, title):
    plt.figure(figsize=(20, 5))

    image = batch[0].numpy()
    color_image = image[:, :, :3]
    depth_image = image[:, :, 3:6]
    motion_vector_image = image[:, :, 6:9]
    motion_blur_image = image[:, :, 9:12]
  
    image_list = [color_image, depth_image, motion_vector_image, motion_blur_image]

    for i in range(len(image_list)):
      plt.subplot(1, 4, i+1)
      plt.title(title[i])
      plt.imshow(image_list[i])
      plt.axis('off')
    plt.show()
    
def generate_images(model, test_input, tar, img_name):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.savefig('generated_images/'+ img_name)
