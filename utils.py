import tensorflow as tf


def activate_gpu_tf(logger=None):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        if logger:
            logger.info('GPU activated')
