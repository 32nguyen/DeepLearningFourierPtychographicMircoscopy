from config import *
from alpha_blending import alpha_bledning_full_patch, alpha_blending_batch_region
from keras.models import load_model
import matplotlib.pyplot as plt
import time
from libtiff import TIFF
def box_type(row, col, row_sub, col_sub):
    if row == 0:
        if col == 0:
            types = 'conner1'
        elif col == len(col_sub) - 1:
            types = 'conner2'
        else:
            types = 'top'
    elif row == len(row_sub) - 1:
        if col == 0:
            types = 'conner4'
        elif col == len(col_sub) - 1:
            types = 'conner3'
        else:
            types = 'bot'
    elif col == 0:
        types = 'left'
    elif col == len(col_sub) - 1:
        types = 'right'
    else:
        types = 'center'
    return types

row_b = 2160
col_b = 2560
subregion = 16
name = "Hela" ## Hela, MCF10A, U2OS_stained, U2OS_unstained
#name = "MCF10A"
#name = "U2OS_stained"
#name = "U2OS_unstained"

final_FOV_predict = name + 'final_FOV_predict_DenseBF9DF20'


# ========================Initialization====================================================
print('FPM CNN reconstruction')
print('Loading Model and Weights')

# GENERATOR
# Our generator is a DenseNet
# ----------------------
generator = load_model('./models_weights/gen_BF9DF20_model.h5')


# loading weights
generator.load_weights('./models_weights/gen_' + name + '_BF9DF20_Floss.h5')


row_sub = np.arange(0, (80 - 19) * 11, 80 - 19)
col_sub = np.arange(0, (80 - 15) * 13, 80 - 15)
FOV_row_sub = np.arange(0, (320 - 76) * 11, 320 - 76)
FOV_col_sub = np.arange(0, (320 - 60) * 13, 320 - 60)

# alpha batch block - small region
pix_blend_r = 76
pix_blend_c = 60
alpha_box = alpha_blending_batch_region(pix_blend_r, pix_blend_c, shape=(4, 320, 320))

# alpha batch block - big region
pix_blend_R = 80
pix_blend_C = 320
alpha_BOX = alpha_blending_batch_region(pix_blend_R, pix_blend_C, shape=(1, 2760, 3440))
FOV_row = np.arange(0, (2760 - pix_blend_R) * 4, 2760 - pix_blend_R)
FOV_col = np.arange(0, (3440 - pix_blend_C) * 4, 3440 - pix_blend_C)

# ========================FPM CNN reconstruction====================================================
start = time.time()


ILED = np.load("./Data/ILED_" + name + "_BF9.npy")
ILED1 = np.load("./Data/ILED_" + name + "_DF20.npy")
ILED = np.concatenate((ILED, ILED1), axis=-1)

FOV_sub = np.zeros(shape=(16, 2760, 3440), dtype=np.float32)

for ind in range(4):
    ILED_temp = ILED[ind * 4:(ind + 1) * 4]
    for row in range(len(row_sub)):
        for col in range(len(col_sub)):
            ILED_sub = ILED_temp[:, row_sub[row]: row_sub[row] + 80,
                       col_sub[col]: col_sub[col] + 80, :]
            predic_P = generator.predict(ILED_sub)[:, :, :, 0]
            type_box1 = box_type(row, col, row_sub, col_sub)
            a = alpha_bledning_full_patch(alpha_box, pix_blend_r, pix_blend_c, type_box1, shape=(4, 320, 320))
            FOV_sub[ind * 4:(ind + 1) * 4, FOV_row_sub[row]: FOV_row_sub[row] + 320,
                    FOV_col_sub[col]: FOV_col_sub[col] + 320] += predic_P * a

final_FOV = np.zeros(shape=(10800, 12800), dtype=np.float32)

for row in range(len(FOV_row)):
    for col in range(len(FOV_col)):
        types_box2 = box_type(row, col, FOV_row, FOV_col)
        a = alpha_bledning_full_patch(alpha_BOX, pix_blend_R, pix_blend_C, types_box2, shape=(1, 2760, 3440))
        temp = FOV_sub[row * 4 + col] * a
        final_FOV[FOV_row[row]: FOV_row[row] + 2760, FOV_col[col]: FOV_col[col] + 3440] += temp[0]
        final_FOV[np.where(final_FOV < 0)] = 0.0
        final_FOV[np.where(final_FOV > 1)] = 1.0
print('Finsihed reconstruction for this frame in Time: %s' % (time.time() - start))
print('saving FPM CNN reconstructed phase', time.time() - start)
plt.imshow(final_FOV, cmap='gray')
plt.show()
