import tensorflow as tf
import DataLoader

num_classes = 2
model = tf.keras.models.load_model("G:\AI\BinaryImageSegmentation\model.h5")
image = tf.io.read_file(r'G:\AI\BinaryImageSegmentation\Data\train\images\CFD_002.jpg')
image = tf.io.decode_jpeg(image)
image = tf.image.resize(image, [128,128])
image = image/255

mask = tf.io.read_file(r'G:\AI\BinaryImageSegmentation\Data\train\masks\CFD_002.jpg')
mask = tf.io.decode_jpeg(mask)
mask = tf.image.resize(mask, [128,128])
mask = mask/255


y_pred= model.predict(image[None,...])[0]
DataLoader.display([image,mask, y_pred])
