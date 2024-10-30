def set_log_level(level):
    from os import environ
    environ['TF_CPP_MIN_LOG_LEVEL'] =  str(int(level))

# Handle GPU selection
def set_gpu(gpu_id):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            if gpu_id is None:
                tf.config.experimental.set_visible_devices([], 'GPU')
            else:
                tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
                tf.config.experimental.set_visible_devices(gpus[gpu_id], 'GPU')
        except RuntimeError as e:
            # Visible devices must be set at program startup
            print(e)

