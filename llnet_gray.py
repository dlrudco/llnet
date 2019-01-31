import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import read_csv
from sklearn.model_selection import train_test_split
from PIL import Image
import imageutil as iu
import time

def pause():
    programPause = input("Press the <ENTER> key to continue...")
def loadcsv(data):
    raw_data = read_csv(data)
    return raw_data.values[:,1:]

n_hidden = [2000, 1600, 1200]; 

n_input =   289;
datasize = 8370;

# orig_test = loadcsv("gray_original_test.csv")
# print("orig_test load complete")
# test = loadcsv("gray_dark_test.csv")
# print("dark_data load complete")
# test = loadcsv("gray_noise_test.csv")
# print("nois_data load complete")
# test = loadcsv("gray_combine_test.csv")
# print("comb_data load complete")


# for i in range (datasize):
#     if i%1000 == 0:
#         print(i)
#     filename_o = 'test_o/' + str(1+i) + '.jpg';
#     filename_d = 'test_d/' + str(1+i) + '.jpg';
#     filename_n = 'test_n/' + str(1+i) + '.jpg';
#     filename_c = 'test_c/' + str(1+i) + '.jpg';

#     img_o = Image.open( filename_o ).convert('L')
#     img_d = Image.open( filename_d ).convert('L')
#     img_n = Image.open( filename_n ).convert('L')
#     img_c = Image.open( filename_c ).convert('L')

#     try:
#         temp_o = np.asarray( img_o, dtype='uint8' )
#     except SystemError:
#         temp_o = np.asarray( img_o.getdata(), dtype='uint8' )

#     try:
#         temp_d = np.asarray( img_d, dtype='uint8' )
#     except SystemError:
#         temp_d = np.asarray( img_d.getdata(), dtype='uint8' )

#     try:
#         temp_n = np.asarray( img_n, dtype='uint8' )
#     except SystemError:
#         temp_n = np.asarray( img_n.getdata(), dtype='uint8' )

#     try:
#         temp_c = np.asarray( img_c, dtype='uint8' )
#     except SystemError:
#         temp_c = np.asarray( img_c.getdata(), dtype='uint8' )

#     temp_o = temp_o.ravel()
#     temp_d = temp_d.ravel()
#     temp_n = temp_n.ravel()
#     temp_c = temp_c.ravel()


#     orig_data[i,:] = np.true_divide(temp_o,255.);
#     dark_data[i,:] = np.true_divide(temp_d,255.);
#     nois_data[i,:] = np.true_divide(temp_n,255.);
#     comb_data[i,:] = np.true_divide(temp_c,255.);


# print("Data Load Complete\n")

X = tf.placeholder(tf.float32, [None, n_input])
ORG = tf.placeholder(tf.float32, [None, n_input])
X_l1 = tf.placeholder(tf.float32, [None, n_hidden[0]])
ORG_l1 = tf.placeholder(tf.float32, [None, n_hidden[0]])
X_l2 = tf.placeholder(tf.float32, [None, n_hidden[1]])
ORG_l2 = tf.placeholder(tf.float32, [None, n_hidden[1]])
###########Dropout rate set##################
#keeprate = tf.placeholder(tf.float32)
####################################
###########Layer set#########################
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden[0]]))
b_encode = tf.Variable(tf.random_normal([n_hidden[0]]))

W_encode1 = tf.Variable(tf.random_normal([n_hidden[0], n_hidden[1]]))
b_encode1 = tf.Variable(tf.random_normal([n_hidden[1]]))

W_encode2 = tf.Variable(tf.random_normal([n_hidden[1], n_hidden[2]]))
b_encode2 = tf.Variable(tf.random_normal([n_hidden[2]]))

W_decode2 = tf.Variable(tf.random_normal([n_hidden[2], n_hidden[1]]))
#W_decode2 = tf.transpose(W_encode2)
b_decode2 = tf.Variable(tf.random_normal([n_hidden[1]]))

W_decode1 = tf.Variable(tf.random_normal([n_hidden[1], n_hidden[0]]))
#W_decode1 = tf.transpose(W_encode1)
b_decode1 = tf.Variable(tf.random_normal([n_hidden[0]]))

W_decode = tf.Variable(tf.random_normal([n_hidden[0], n_input]))
#W_decode = tf.transpose(W_encode)
b_decode = tf.Variable(tf.random_normal([n_input]))

########################################################################
#Input
#
encoder = tf.nn.sigmoid(
                tf.add(tf.matmul(X, W_encode), b_encode))
#encoder = tf.nn.dropout(encoder,keeprate);

encoder1 = tf.nn.sigmoid(
                tf.add(tf.matmul(encoder, W_encode1), b_encode1))
#encoder1 = tf.nn.dropout(encoder1,keeprate);

encoder2 = tf.nn.sigmoid(
                tf.add(tf.matmul(encoder1, W_encode2), b_encode2))
#encoder2 = tf.nn.dropout(encoder2,keeprate);

decoder2 = tf.nn.sigmoid(
                tf.add(tf.matmul(encoder2, W_decode2), b_decode2))
#decoder2 = tf.nn.dropout(decoder2,keeprate);
decoder1 = tf.nn.sigmoid(
                tf.add(tf.matmul(decoder2, W_decode1), b_decode1))
#decoder1 = tf.nn.dropout(decoder1,keeprate);

decoder = tf.nn.sigmoid(
                tf.add(tf.matmul(decoder1, W_decode), b_decode))
#decoder = tf.nn.dropout(decoder,keeprate);
#Output
decoder_pre = tf.nn.sigmoid(tf.add(tf.matmul(encoder,W_decode),b_decode))
decoder1_pre = tf.nn.sigmoid(tf.add(tf.matmul(encoder1,W_decode1),b_decode1))
decoder_pre_l2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder1_pre,W_decode),b_decode))

SAVER_DIR = ["model_dark_rlb850015000015_gray_test_l1"]
			#,"model_noise_rlb531_gray","model_combine_rlb531_gray"]

for path in SAVER_DIR:
    
    
    
    saver = tf.train.Saver()
    rgbpath = path;
    ckpt_path = os.path.join(rgbpath,"model")
    ckpt = tf.train.get_checkpoint_state(rgbpath)

    sess = tf.Session()
    start_time = time.time()
    #orig_test = iu.split("normal_car.jpg")
    #print("orig_test load complete")
    test = iu.split("dark_car.jpg")
    print("dark_test load complete")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model  " + rgbpath + "  Load Complete")
    else:
        print('no trained model! Train model first')
        break;



    modified = sess.run(decoder_pre,
               feed_dict={X:test})
    # mod_l1 = sess.run(decoder_pre,
    #             feed_dict={X:orig_test})
    # weight = np.zeros((200,419));
    # nimg = np.zeros((200,419));
    # oimg = np.zeros((200,419));
    # temp = np.ones((17,17))
    # for i in range(8370) :   
    #     arr = np.multiply(modified[i,:],255.).reshape(17,17);
    #     oarr = np.multiply(mod_l1[i,:],255.).reshape(17,17)
    #     if i%1000 == 0:
    #         print(i)
    #         print(arr.astype(np.uint8))
    #     else: None
    #     row = i / 135
    #     col = i % 135
    #     weight[3*row:3*row+17,3*col:3*col+17] += temp;
    #     nimg[3*row:3*row+17,3*col:3*col+17] += arr;
    #     oimg[3*row:3*row+17,3*col:3*col+17] += oarr;

    # nimg = np.divide(nimg,weight).astype(np.uint8);
    # result = Image.fromarray(nimg.astype(np.uint8), 'L')
    # oimg = np.divide(oimg,weight).astype(np.uint8);
    # oresult = Image.fromarray(oimg.astype(np.uint8), 'L')
    # oresult.show("original")
    
    result = iu.stitch(modified)
    print("---Total process %s seconds ---" % (time.time() - start_time))
    result.show()
    if path[6] == "d" :
        result.save('recon_dark_car.jpg', 'JPEG')
    elif path[6] == "n" :
        result.save('recon_noise_car.jpg', 'JPEG')
    elif path[6] == "c" :
        result.save('recon_combine_car.jpg', 'JPEG')
    result.show("reconstructed")
        
