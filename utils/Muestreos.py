import tensorflow as tf

def A_Fran(y, Masks):
    mul_meas = tf.multiply(tf.math.conj(Masks), y)
    return  tf.signal.fft2d(mul_meas)

def AT_Fran(y, Masks):
    
    mult_mass_z = tf.multiply(Masks, tf.signal.ifft2d(y))
    res = tf.reduce_sum(mult_mass_z, axis=1, keepdims=True)
    return tf.multiply(res, tf.cast(y.shape[2]*y.shape[3], dtype=res.dtype))
