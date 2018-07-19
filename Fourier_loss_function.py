import tensorflow as tf

def fft_abs_for_map_fn(x):
    #x = (x + 1.) / 2.
    # the label expected in range 0 - 1
    x_complex = tf.complex(x, tf.zeros_like(x))[:, :, 0]
    fft = tf.spectral.fft2d(x_complex)
    fft_abs = tf.abs(fft)
    return fft_abs

'''def Fourier_Loss(y_true, y_pred):
    # TODO:
    print(y_true)
    FFT_pred = tf.fft2d(tf.squeeze(tf.cast(y_pred, dtype=tf.complex64)))
    FFT_true = tf.fft2d(tf.squeeze(tf.cast(y_true, dtype=tf.complex64)))

    Amp_FFT_pred = tf.abs(FFT_pred)
    Ang_FFT_pred = tf.angle(FFT_pred)

    Amp_FFT_true = tf.abs(FFT_true)
    Ang_FFT_true = tf.angle(FFT_true)

    #F_MAE = tf.reduce_mean(tf.abs(Amp_FFT_pred-Amp_FFT_true)) + \
    #      tf.reduce_mean(tf.abs(Ang_FFT_pred-Ang_FFT_true))
    F_MAE = tf.reduce_mean(tf.abs(Amp_FFT_pred - Amp_FFT_true))
    MAE = tf.reduce_mean(tf.abs(y_pred-y_true))
    F_MAE = tf.cast(F_MAE, dtype=tf.float32)
    # return mean absolute error
    return 0.05*F_MAE + 0.95*MAE'''

def Fourier_Loss(y_true, y_pred):
    # TODO:
    FFT_pred = tf.map_fn(fft_abs_for_map_fn, y_pred)
    FFT_true = tf.map_fn(fft_abs_for_map_fn, y_true)

    F_MAE = tf.reduce_mean(tf.abs(FFT_pred - FFT_true))
    MAE = tf.reduce_mean(tf.abs(y_pred-y_true))
    F_MAE = tf.cast(F_MAE, dtype=tf.float32)
    # return mean absolute error
    return 0.05*F_MAE + 0.95*MAE