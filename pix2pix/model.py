import tensorflow as tf

def Generator():
  inputs = tf.keras.layers.Input(shape=[256, 256, 18])

  down_stack = [
    downsample(64, 4, apply_batchnorm=False), #(128, 128, 64)
    downsample(128, 4),                       #(64, 64, 128)
    downsample(256, 4),                       #(16, 16, 256)
    downsample(512, 4),                       #(32, 32, 512) 
    downsample(512, 4),                       #(8, 8, 512)
    downsample(512, 4),                       #(4, 4, 512)
    downsample(512, 4),                       #(2, 2, 512) 
    downsample(512, 4),                       #(1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),    #(2, 2, 512)
    upsample(512, 4, apply_dropout=True),    #(4, 4, 512)
    upsample(512, 4, apply_dropout=True),    #(8, 8, 512)
    upsample(512, 4),                        #(16, 16, 512)
    upsample(256, 4),                        #(32, 32, 256)
    upsample(128, 4),                        #(64, 64, 128)
    upsample(64, 4),                         #(128, 128, 64)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(3, 4,
                                        strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        activation='tanh') 

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # Mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (100 * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[256, 256, 21], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])

  down1 = downsample(64, 4, False)(x)                         #(128, 128, 64)
  print(down1.shape)
  down2 = downsample(128, 4)(down1)                           #(64, 64, 128)
  print(down2.shape)                           
  down3 = downsample(256, 4)(down2)                           #(32, 32, 256)
  print(down3.shape)

  conv1 = tf.keras.layers.Conv2D(256, 4, strides=1, padding='same',
                                  kernel_initializer=initializer, use_bias=False)(down3) #(32, 32, 256)
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv1)                               #(32, 32, 256)
  leaky_relu1 = tf.keras.layers.LeakyReLU()(batchnorm1)                                  #(32, 32, 256)
  zero_pad1 = tf.keras.layers.ZeroPadding2D()(leaky_relu1)                               #(34, 34, 256)

  conv2 = tf.keras.layers.Conv2D(512, 4, strides=1,                                      #(31, 31, 512)
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)
  
  batchnorm2 = tf.keras.layers.BatchNormalization()(conv2)                               #(31, 31, 512)
  leaky_relu2 = tf.keras.layers.LeakyReLU()(batchnorm2)                                  #(31, 31, 512)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu2)                               #(33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)               #(30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
  loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                            kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result