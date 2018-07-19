import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from scipy import ndimage
from config import *
import scipy.ndimage

def normalize_uniform(X):
    X = np.ndarray.astype(X, dtype=np.float32)
    X -= np.min(X)
    X /= np.max(X)
    return X

def normalize_mean_std(X):
    X = np.ndarray.astype(X, dtype=np.float32)
    ave = np.mean(X)
    std = np.std(X)
    X -= ave
    X /= std
    return X

def high_pass(image, pixel):
    lowpass = ndimage.gaussian_filter(image, pixel)
    gauss_highpass = image - lowpass
    return gauss_highpass
'''
def data_generator():
    ILED = np.load('ILED.npy')  # (2160, 2560, 195)
    plt.imshow(ILED[:,:,65], cmap='gray')
    plt.show()
    I = np.load('intensity_giga_img.npy')  # (10800, 12800)
    P = np.load('phase_giga_img.npy')  # (10800, 12800)
    print ILED.shape, I.shape, P.shape
    print batch_size
    # iterate forever bc keras requires this

    batch_features = np.zeros((batch_size, i_in, i_in, ILED.shape[2]), dtype=np.float32)
    batch_labels = np.zeros((batch_size, i_out, i_out, 2), dtype=np.float32)
    print batch_labels.shape, batch_features.shape
    ind_r = np.random.randint(0, int(ILED.shape[0] / i_in), batch_size)
    ind_c = np.random.randint(0, int(ILED.shape[1] / i_in), batch_size)

    print ind_r, ind_c
    for ind in range(0, batch_size):
        ILED_sub = ILED[ind_r[ind] * i_in : (ind_r[ind] + 1) * i_in,
                   ind_c[ind] * i_in : (ind_c[ind] + 1) * i_in, :]
        print ILED_sub.shape, ind_c[ind]
        I_sub = I[ind_r[ind] * i_out : (ind_r[ind] + 1) * i_out,
                ind_c[ind] * i_out : (ind_c[ind] + 1) * i_out]
        P_sub = P[ind_r[ind] * i_out : (ind_r[ind] + 1) * i_out,
                ind_c[ind] * i_out : (ind_c[ind] + 1) * i_out]
        batch_features[ind] = ILED_sub
        batch_labels[ind, :, :, 0] = I_sub
        batch_labels[ind, :, :, 1] = P_sub

    return (batch_features, batch_labels)
'''

'''def generating(ILED, P, train):
    batch_features = np.zeros((params.batch_size, params.i_in, params.i_in, ILED.shape[2]), dtype=np.float32)
    batch_labels = np.zeros((params.batch_size, params.i_out_scale, params.i_out_scale, 1), dtype=np.float32)

    valid_binary = np.zeros((ILED.shape[0], ILED.shape[1]), dtype=np.float32)
    # print batch_labels.shape, batch_features.shape
    start_r = 720
    end_r = 720 * 2
    start_c = 850
    end_c = 850 * 2
    valid_binary[start_r:end_r, start_c:end_c] = 1.0
    if train:
        val_i = params.batch_size
        ind_r = []
        ind_c = []
        while val_i > 0:
            r = np.random.randint(0, ILED.shape[0] - params.i_in, 1)
            c = np.random.randint(0, ILED.shape[1] - params.i_in, 1)
            if valid_binary[r, c] == 1.0:
                continue
            else:
                ind_r.append(r[0])
                ind_c.append(c[0])
                val_i -= 1

    else:  # valid
        ind_r = np.random.randint(start_r, end_r - params.i_in, params.batch_size)
        ind_c = np.random.randint(start_c, end_c - params.i_in, params.batch_size)

    # ind_r = np.random.randint(0, ILED.shape[0] - i_in, batch_size)
    # ind_c = np.random.randint(0, ILED.shape[1] - i_in, batch_size)

    for ind in range(0, params.batch_size):
        # print ind
        ILED_sub = ILED[ind_r[ind]: ind_r[ind] + params.i_in,
                   ind_c[ind]: ind_c[ind] + params.i_in, :]
        ILED_sub = high_pass(ILED_sub)  # train on high frequency image
        # print ILED_sub.shape, ind_c[ind]
        # I_sub = I[ind_r[ind] * 5 : ind_r[ind] * 5 + params.i_in*5,
        #          ind_c[ind] * 5 : ind_c[ind] * 5 + params.i_in*5]
        P_sub = P[ind_r[ind] * 5: ind_r[ind] * 5 + params.i_in * 5,
                ind_c[ind] * 5: ind_c[ind] * 5 + params.i_in * 5]
        # I_sub = scipy.ndimage.zoom(I_sub, params.i_out_scale * 1.0 / I_sub.shape[0], order=0)
        P_sub = scipy.ndimage.zoom(P_sub, params.i_out_scale * 1.0 / P_sub.shape[0], order=0)
        batch_features[ind] = ILED_sub
        # batch_labels[ind, :, :, 0] = I_sub
        batch_labels[ind, :, :, 0] = P_sub
    return batch_features, batch_labels

def data_generator(train=True):
    ILED = np.load('ILED.npy')  # (2160, 2560, 195)
    #plt.imshow(ILED[:,:,65], cmap='gray')
    #plt.show()
    #I = np.load('intensity_giga_img.npy')  # (10800, 12800)
    P = np.load('phase_giga_img.npy')  # (10800, 12800)
    #I = normalize_uniform(I)
    P = normalize_uniform(P)
    #print ILED.shape, I.shape, P.shape
    #print batch_size
    # iterate forever bc keras requires this

    while True:
        batch_features, batch_labels = generating(ILED, P, train=train)
        yield (batch_features, batch_labels)
    #batch_features, batch_labels = generating(ILED, P, train=train)
    #return batch_features, batch_labels'''
def scale_input(batch_image, i_out_scale):
    S = batch_image.shape
    image = np.ndarray(shape=(S[0], i_out_scale, i_out_scale, S[3]), dtype=np.float32)
    for i in range(S[0]):
        for j in range(S[3]):
            M = batch_image[i,:,:,j]
            image[i,:,:,j] = ndimage.zoom(M, i_out_scale * 1.0 / M.shape[1], order=0)
    return image

def generating_subregion(ILED, num_region):

    batch_features = np.zeros((params.batch_size, params.i_in, params.i_in, ILED.shape[3]), dtype=np.float32)
    batch_labels = np.zeros((params.batch_size, params.i_out_scale, params.i_out_scale, 1), dtype=np.float32)
    n_batch_re = int(params.batch_size/num_region)

    for ind_region in range(0, num_region):
        for i in range(0, n_batch_re):
            ind_r = np.random.randint(0, ILED.shape[1] - params.i_in)
            ind_c = np.random.randint(0, ILED.shape[2] - params.i_in)
            ILED_sub = ILED[ind_region,
                            ind_r: ind_r + params.i_in,
                            ind_c: ind_c + params.i_in, :]

            #P_sub = P[ind_region,
            #          ind_r * 4: (ind_r + params.i_in) * 4,
            #          ind_c * 4: (ind_c + params.i_in) * 4, :]

            #P_sub = high_pass(P_sub, 5)
            #P_sub = normalize_uniform(P_sub)
            #P_sub = scipy.ndimage.zoom(P_sub, params.i_out_scale * 1.0 / P_sub.shape[1], order=0)
            batch_features[ind_region*n_batch_re + i] = ILED_sub
            batch_labels[ind_region*n_batch_re + i] = 0

    batch_features_up = scale_input(batch_features, params.i_out_scale)
    return (batch_features, batch_features_up, batch_labels)


def data_generator_subregion(num_region, train=True):
    if train:
        ILED = np.load('ILED_Hela_BF13_0700.npy')  # (16, 675, 800, 13)
        ILED1 = np.load('ILED_Hela_DF10_0700.npy')  # (16, 675, 800, 10)
        #ILED1 = np.load('ILED_Hela_DF36_0700.npy')  # (16, 675, 800, 10)
        ILED = np.concatenate((ILED, ILED1), axis=-1)  # (16, 2160, 2560, 23)
        #P = np.load('phase_Hela_0700.npy')  # (16, 675, 800, 1)
    else:
        ILED = np.load('ILED_Hela_BF13_1100.npy')  # (16, 2160, 2560, 13)
        ILED1 = np.load('ILED_Hela_DF10_1100.npy')  # (16, 2160, 2560, 10)
        #ILED1 = np.load('ILED_Hela_DF36_1100.npy')  # (16, 2160, 2560, 10)
        ILED = np.concatenate((ILED, ILED1), axis=-1)  # (16, 2160, 2560, 23)
        P = np.load('phase_Hela_1100.npy')  # (16, 2700, 3200, 1)
    # iterate forever bc keras requires this
    while True:
        batch_features, batch_features_up, batch_labels = generating_subregion(ILED, num_region)
        yield batch_features, batch_features_up, batch_labels
    #batch_features, batch_labels, batch_features_up = generating_subregion(ILED, P, num_region)
    #return batch_features, batch_labels, batch_features_up

"""batch_features, batch_labels = data_generator_subregion(16, train=False)
print batch_features.shape, batch_labels.shape
for x in range(2):
    for y in range(6):
        plt.subplot(7, 2, x*2+y+1)
        plt.imshow(batch_features[0, :, :, x*2+y], cmap='gray')
plt.subplot(7, 2, 14)
plt.imshow(batch_labels[0,:,:,0], cmap='gray')

plt.show()"""

'''batch_features, batch_labels, batch_features_up = data_generator_subregion(16, train=False)
print batch_features.shape, batch_labels.shape, batch_features_up.shape
print np.min(batch_features), np.min(batch_features_up)
print np.max(batch_features), np.max(batch_features_up)
for x in range(2):
    for y in range(6):
        plt.subplot(7, 2, x*2+y+1)
        plt.imshow(batch_features[0, :, :, x*2+y], cmap='gray')
plt.subplot(7, 2, 13)
plt.imshow(batch_labels[0,:,:,0], cmap='gray')
plt.subplot(7, 2, 14)
plt.imshow(batch_features_up[0,:,:,0], cmap='gray')
plt.show()'''