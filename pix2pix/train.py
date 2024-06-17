import tensorflow as tf

import os
import time
import datetime
import util
import model

from matplotlib import pyplot as plt
from IPython import display

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BUFFER_SIZE = 400
BATCH_SIZE = 1
PATH = './datasets/'

color_files = tf.data.Dataset.list_files(str(PATH + 'color_1layer/*.png'), shuffle=False)
depth_files = tf.data.Dataset.list_files(str(PATH + 'depth_1layer/*.png'), shuffle=False)
motion_vector_files = tf.data.Dataset.list_files(str(PATH + 'motion_vector_1layer/*.png'), shuffle=False)
motion_blur_files = tf.data.Dataset.list_files(str(PATH + 'motion_blur/*.png'), shuffle=False)
dataset = tf.data.Dataset.zip((color_files, depth_files, motion_vector_files, motion_blur_files))
dataset = dataset.map(lambda color, depth, motion_vector, motion_blur: util.parse_function(color, depth, motion_vector, motion_blur),
                      num_parallel_calls=tf.data.AUTOTUNE)

train_dataset = dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

for batch in train_dataset.take(1):
    util.display_images(batch, ['color image', 'depth image', 'motion vector image', 'motion blur image'])

total_images = 0
for batch in train_dataset:
    total_images += batch.shape[0]
print(f'Total images in dataset: {total_images}')

for batch in train_dataset.take(1):
  down_model = model.downsample(3, 4)
  image = tf.expand_dims(batch[0].numpy()[:, :, :], 0)
  down_result = down_model(image, 0)
  print (f'Image shape: {batch.shape} to down result shape: {down_result.shape}')

for batch in train_dataset.take(1):
  up_model = model.upsample(3, 4)
  up_result = up_model(down_result)
  print (f'down result shape: {down_result.shape} to Up result shape: {up_result.shape}')

generator = model.Generator()
for batch in train_dataset.take(1):
  image = batch[0].numpy()[:, :, :]
  gen_output = generator(image[tf.newaxis, ...], training=False)
  plt.imshow(gen_output[0, ...])

discriminator = model.Discriminator()
for batch in train_dataset.take(1):
  image = batch[0].numpy()[:, :, :]
  disc_out = discriminator([image[tf.newaxis, ...], gen_output], training=False)
  plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
  plt.colorbar()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, step):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = model.generator_loss(disc_generated_output, gen_output, target)
    disc_loss = model.discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

def fit(train_ds, test_ds, steps):
  example_input, example_target = next(iter(test_ds.take(1)))
  start = time.time()

  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    if (step) % 1000 == 0:
      display.clear_output(wait=True)

      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      start = time.time()

      generate_images(generator, example_input, example_target, str(step))
      print(f"Step: {step//1000}k")

    train_step(input_image, target, step)

    # Training step
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)

    # Save (checkpoint) the model every 5k steps
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

fit(train_dataset, test_dataset, steps=40000)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

for inp, tar in test_dataset.take(5):
  generate_images(generator, inp, tar)