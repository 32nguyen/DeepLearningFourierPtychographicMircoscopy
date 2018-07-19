import numpy as np
import matplotlib.pyplot as plt
def alpha_blending(img1, img2, pix_blend):
    S1 = img1.shape
    S2 = img2.shape
    row_img1 = np.ones((1, S1[2]), dtype=np.float32)
    row_img2 = np.ones((1, S2[2]), dtype=np.float32)
    alpha_line1 = np.linspace(1, 0, pix_blend)
    alpha_line2 = np.linspace(0, 1, pix_blend)
    row_img1[0,-pix_blend:] = alpha_line1
    row_img2[0,:pix_blend] = alpha_line2
    alpha_img1 = np.repeat(row_img1, S1[1], axis=0)
    alpha_img2 = np.repeat(row_img2, S1[1], axis=0)
    alpha_img1 = np.expand_dims(alpha_img1, axis=0)
    alpha_img2 = np.expand_dims(alpha_img2, axis=0)
    alpha_img1 = np.repeat(alpha_img1, S1[0], axis=0)
    alpha_img2 = np.repeat(alpha_img2, S2[0], axis=0)
    img_out1 = np.zeros((S1[0], S1[1], S1[2] + S2[2] - pix_blend), dtype=np.float32)
    img_out2 = np.zeros((S1[0], S1[1], S1[2] + S2[2] - pix_blend), dtype=np.float32)
    img_out1[:, :, :S1[2]] = alpha_img1 * img1
    img_out2[:, :, S1[2]-pix_blend:] = alpha_img2 * img2
    return alpha_img1, alpha_img2, img_out1+img_out2

def alpha_blending_batch_region(pix_blend_r, pix_blend_c, shape=(16, 320, 320)):
    S = shape
    alpha_line_r = np.expand_dims(np.expand_dims(np.linspace(1, 0, pix_blend_r), axis=1), axis=0)
    alpha_line_c = np.expand_dims(np.expand_dims(np.linspace(1, 0, pix_blend_c), axis=0), axis=0)


    al_right = np.repeat(np.repeat(alpha_line_c, S[1], axis=1), S[0], axis=0)
    al_left = np.rot90(al_right, 2, axes=(1, 2))
    al_bot = np.repeat(np.repeat(alpha_line_r, S[2], axis=2), S[0], axis=0)
    #print(al_bot.shape)
    al_top = np.rot90(al_bot, 2, axes=(1, 2))


    #al_c = np.repeat(np.repeat(alpha_line_c*0.5, pix_blend_c, axis=1), S[0], axis=0)
    #al_r = np.repeat(np.repeat(alpha_line_c * 0.5, pix_blend_r, axis=1), S[0], axis=0)
    al_c_righttop = np.repeat(np.repeat(alpha_line_c*0.5, pix_blend_r, axis=1), S[0], axis=0)
    al_c_rightbot = np.repeat(np.repeat(alpha_line_r*0.5, pix_blend_c, axis=2), S[0], axis=0)
    al_c_leftbot = np.rot90(al_c_righttop, 2, axes=(1, 2))
    al_c_lefttop= np.rot90(al_c_rightbot, 2, axes=(1, 2))

    al_c_righttop + al_c_rightbot + al_c_leftbot + al_c_lefttop
    return (al_right, al_left, al_bot, al_top, al_c_righttop, al_c_rightbot, al_c_leftbot, al_c_lefttop)

def alpha_bledning_full_patch(alpha_box, pix_blend_r, pix_blend_c, types='center', shape=(16, 320, 320)):
    alpha_patch = np.ones(shape=shape, dtype=np.float32)
    (al_right, al_left, al_bot, al_top, al_c_righttop, al_c_rightbot, al_c_leftbot, al_c_lefttop) = alpha_box
    if types=='center':
        alpha_patch[:, :, -pix_blend_c:] = al_right
        alpha_patch[:, :, 0:pix_blend_c] = al_left
        alpha_patch[:, -pix_blend_r:, :] = al_bot
        alpha_patch[:, 0:pix_blend_r, :] = al_top

        alpha_patch[:, 0:pix_blend_r:, -pix_blend_c:] = al_c_righttop
        alpha_patch[:, 0:pix_blend_r:, 0:pix_blend_c] = al_c_lefttop
        alpha_patch[:, -pix_blend_r:, -pix_blend_c:] = al_c_rightbot
        alpha_patch[:, -pix_blend_r:, 0:pix_blend_c] = al_c_leftbot
    elif types=='conner1':
        alpha_patch[:, :, -pix_blend_c:] = al_right
        alpha_patch[:, -pix_blend_r:, :] = al_bot
        alpha_patch[:, -pix_blend_r:, -pix_blend_c:] = al_c_rightbot
    elif types=='conner2':
        alpha_patch[:, :, 0:pix_blend_c] = al_left
        alpha_patch[:, -pix_blend_r:, :] = al_bot
        alpha_patch[:, -pix_blend_r:, 0:pix_blend_c] = al_c_leftbot
    elif types=='conner3':
        alpha_patch[:, :, 0:pix_blend_c] = al_left
        alpha_patch[:, 0:pix_blend_r, :] = al_top
        alpha_patch[:, 0:pix_blend_r:, 0:pix_blend_c] = al_c_lefttop
    elif types=='conner4':
        alpha_patch[:, :, -pix_blend_c:] = al_right
        alpha_patch[:, 0:pix_blend_r, :] = al_top
        alpha_patch[:, 0:pix_blend_r:, -pix_blend_c:] = al_c_righttop
    elif types=='right':
        alpha_patch[:, :, 0:pix_blend_c] = al_left
        alpha_patch[:, -pix_blend_r:, :] = al_bot
        alpha_patch[:, 0:pix_blend_r, :] = al_top

        alpha_patch[:, 0:pix_blend_r:, 0:pix_blend_c] = al_c_lefttop
        alpha_patch[:, -pix_blend_r:, 0:pix_blend_c] = al_c_leftbot

    elif types=='top':
        alpha_patch[:, :, -pix_blend_c:] = al_right
        alpha_patch[:, :, 0:pix_blend_c] = al_left
        alpha_patch[:, -pix_blend_r:, :] = al_bot

        alpha_patch[:, -pix_blend_r:, -pix_blend_c:] = al_c_rightbot
        alpha_patch[:, -pix_blend_r:, 0:pix_blend_c] = al_c_leftbot
    elif types=='left':
        alpha_patch[:, :, -pix_blend_c:] = al_right
        alpha_patch[:, -pix_blend_r:, :] = al_bot
        alpha_patch[:, 0:pix_blend_r, :] = al_top

        alpha_patch[:, 0:pix_blend_r:, -pix_blend_c:] = al_c_righttop
        alpha_patch[:, -pix_blend_r:, -pix_blend_c:] = al_c_rightbot
    elif types=='bot':
        alpha_patch[:, :, -pix_blend_c:] = al_right
        alpha_patch[:, :, 0:pix_blend_c] = al_left
        alpha_patch[:, 0:pix_blend_r, :] = al_top

        alpha_patch[:, 0:pix_blend_r:, -pix_blend_c:] = al_c_righttop
        alpha_patch[:, 0:pix_blend_r:, 0:pix_blend_c] = al_c_lefttop

    return alpha_patch


#pix_blend_r=76
#pix_blend_c=60
#img1 = np.ones(shape=(16, 320, 320), dtype=np.float32)
#al_right, al_left, al_bot, al_top, al_c_righttop, al_c_rightbot, \
#    al_c_leftbot, al_c_lefttop = alpha_blending_batch_region(pix_blend_r, pix_blend_c)

#a = alpha_blending_batch_region(img1, pix_blend_r=100, pix_blend_c=80)
#a = alpha_bledning_full_patch(img1, al_right, al_left, al_bot, al_top, al_c_righttop,
#                              al_c_rightbot, al_c_leftbot, al_c_lefttop,
#                              pix_blend_r, pix_blend_c, types='bot')
#plt.imshow(a[14])
#plt.show()
'''img1 = np.ones(shape=(16, 256, 512), dtype=np.float32)
img2 = np.ones(shape=(16, 256, 256), dtype=np.float32)
a, b, img_out = alpha_blending(img1, img2, 80)
plt.subplot(1,3,1)
plt.imshow(a[1])
plt.subplot(1,3,2)
plt.imshow(b[1])
plt.subplot(1,3,3)
plt.imshow(img_out[1])
plt.show()'''


