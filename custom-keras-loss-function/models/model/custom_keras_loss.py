import tensorflow as tf

def custom_mae(y_true,y_pred):
    return tf.math.abs(tf.math.subtract(tf.cast(y_true,dtype=tf.float32),y_pred))