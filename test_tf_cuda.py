import tensorflow as tf
print("TF Version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("GPU Available:", tf.test.is_gpu_available())

# Test GPU computation
if tf.test.is_gpu_available():
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0])
        b = tf.constant([3.0, 4.0])
        c = tf.add(a, b)
    print("GPU computation test passed")