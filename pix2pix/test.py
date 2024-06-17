from PIL import Image
import numpy as np

image = Image.open('2.bmp')
image_data = np.array(image)

if image_data.shape[2] == 4:
    R, G, B, W = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2], image_data[:, :, 3]
    print("Red channel:", R)
    print("Green channel:", G)
    print("Blue channel:", B)
    print("White channel:", W)

elif image_data.shape[2] == 3:
    R, G, B = image_data[:, :, 0], image_data[:, :, 1], image_data[:, :, 2]
    print("Red channel:", R)
    print("Green channel:", G)
    print("Blue channel:", B)

else :
    R = image_data[:, :, 0]
    print("Red channel:", R)
