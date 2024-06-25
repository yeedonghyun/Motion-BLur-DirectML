import tensorflow as tf
from matplotlib import pyplot as plt

def process_image(image, channels=3):
  image = tf.image.decode_png(image, channels=channels)
  image = tf.image.convert_image_dtype(image, tf.float32)

  return image

def parse_function(paths):
  images = [process_image(tf.io.read_file(path)) for path in paths]
  combined_image = tf.concat(images, axis=-1)
  return combined_image

def display_images(images, title, img_name = None, isSave = False):
  image_list = []
  num_imgaes = len(title)

  for i in range(num_imgaes):
    image_list.append(images[:, :, i * 3:i * 3 + 3])

  plt.figure(figsize=(20, 5))
  for i in range(num_imgaes):
    plt.subplot(1, num_imgaes, i+1)
    plt.title(title[i])
    plt.imshow(image_list[i])
    #plt.imshow(image_list[i] * 0.5 + 0.5)
    plt.axis('off')
  
  if isSave:
    plt.savefig('generated_images/'+ img_name)
  else : 
    plt.show()
    
def generate_images(model, test_input, tar, title, img_name):
  prediction = model(test_input, training=True)
  display_images(test_input + tar[0] + prediction[0], title + ['Generated image'], img_name, True)