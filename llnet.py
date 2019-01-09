import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import read_csv
from sklearn.model_selection import train_test_split
from PIL import Image

n_hidden = [2000, 1600, 1200]; 


n_input =   289;
datasize = 8370;

orig_data=np.empty((3,datasize,n_input))
dark_data=np.empty((3,datasize,n_input))
nois_data=np.empty((3,datasize,n_input))
comb_data=np.empty((3,datasize,n_input))

for i in range (datasize):
    if i%1000 == 0:
        print(i)
    filename_o = 'test_o/' + str(1+i) + '.jpg';
    filename_d = 'test_d/' + str(1+i) + '.jpg';
    filename_n = 'test_n/' + str(1+i) + '.jpg';
    filename_c = 'test_c/' + str(1+i) + '.jpg';

    img_o = Image.open( filename_o )
    img_d = Image.open( filename_d )
    img_n = Image.open( filename_n )
    img_c = Image.open( filename_c )

    try:
        temp_o = np.asarray( img_o, dtype='uint8' )
    except SystemError:
        temp_o = np.asarray( img_o.getdata(), dtype='uint8' )

    try:
        temp_d = np.asarray( img_d, dtype='uint8' )
    except SystemError:
        temp_d = np.asarray( img_d.getdata(), dtype='uint8' )

    try:
        temp_n = np.asarray( img_n, dtype='uint8' )
    except SystemError:
        temp_n = np.asarray( img_n.getdata(), dtype='uint8' )

    try:
        temp_c = np.asarray( img_c, dtype='uint8' )
    except SystemError:
        temp_c = np.asarray( img_c.getdata(), dtype='uint8' )

    temp_o_r = temp_o[:,:,0].ravel()
    temp_d_r = temp_d[:,:,0].ravel()
    temp_n_r = temp_n[:,:,0].ravel()
    temp_c_r = temp_c[:,:,0].ravel()

    temp_o_g = temp_o[:,:,1].ravel()
    temp_d_g = temp_d[:,:,1].ravel()
    temp_n_g = temp_n[:,:,1].ravel()
    temp_c_g = temp_c[:,:,1].ravel()

    temp_o_b = temp_o[:,:,2].ravel()
    temp_d_b = temp_d[:,:,2].ravel()
    temp_n_b = temp_n[:,:,2].ravel()
    temp_c_b = temp_c[:,:,2].ravel()

    orig_data[0,i,:] = np.true_divide(temp_o_r,255.);
    dark_data[0,i,:] = np.true_divide(temp_d_r,255.);
    nois_data[0,i,:] = np.true_divide(temp_n_r,255.);
    comb_data[0,i,:] = np.true_divide(temp_c_r,255.);

    orig_data[1,i,:] = np.true_divide(temp_o_g,255.);
    dark_data[1,i,:] = np.true_divide(temp_d_g,255.);
    nois_data[1,i,:] = np.true_divide(temp_n_g,255.);
    comb_data[1,i,:] = np.true_divide(temp_c_g,255.);

    orig_data[2,i,:] = np.true_divide(temp_o_b,255.);
    dark_data[2,i,:] = np.true_divide(temp_d_b,255.);
    nois_data[2,i,:] = np.true_divide(temp_n_b,255.);
    comb_data[2,i,:] = np.true_divide(temp_c_b,255.);

print("Data Load Complete\n")

X = tf.placeholder(tf.float32, [None, n_input])
ORG = tf.placeholder(tf.float32, [None, n_input])
###########Dropout rate set##################
keeprate = tf.placeholder(tf.float32)
####################################
###########Layer set#########################
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden[0]]))
b_encode = tf.Variable(tf.random_normal([n_hidden[0]]))

encoder = tf.nn.sigmoid(
                tf.add(tf.matmul(X, W_encode), b_encode))
encoder = tf.nn.dropout(encoder,keeprate);

W_encode1 = tf.Variable(tf.random_normal([n_hidden[0], n_hidden[1]]))
b_encode1 = tf.Variable(tf.random_normal([n_hidden[1]]))


encoder1 = tf.nn.sigmoid(
                tf.add(tf.matmul(encoder, W_encode1), b_encode1))
encoder1 = tf.nn.dropout(encoder1,keeprate);

W_encode2 = tf.Variable(tf.random_normal([n_hidden[1], n_hidden[2]]))
b_encode2 = tf.Variable(tf.random_normal([n_hidden[2]]))


encoder2 = tf.nn.sigmoid(
                tf.add(tf.matmul(encoder1, W_encode2), b_encode2))
encoder2 = tf.nn.dropout(encoder2,keeprate);

W_decode2 = tf.Variable(tf.random_normal([n_hidden[2], n_hidden[1]]))
b_decode2 = tf.Variable(tf.random_normal([n_hidden[1]]))

decoder2 = tf.nn.sigmoid(
                tf.add(tf.matmul(encoder2, W_decode2), b_decode2))
decoder2 = tf.nn.dropout(decoder2,keeprate);

W_decode1 = tf.Variable(tf.random_normal([n_hidden[1], n_hidden[0]]))
b_decode1 = tf.Variable(tf.random_normal([n_hidden[0]]))

decoder1 = tf.nn.sigmoid(
                tf.add(tf.matmul(decoder2, W_decode1), b_decode1))
decoder1 = tf.nn.dropout(decoder1,keeprate);

W_decode = tf.Variable(tf.random_normal([n_hidden[0], n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))

decoder = tf.nn.sigmoid(
                tf.add(tf.matmul(decoder1, W_decode), b_decode))
decoder = tf.nn.dropout(decoder,keeprate);

SAVER_DIR = ["model_dark_rlb511","model_noise_rlb511","model_combine_rlb511"]

for path in SAVER_DIR:
    saver = tf.train.Saver()
    ckpt_path = os.path.join(path,"model")
    ckpt = tf.train.get_checkpoint_state(path)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    earlystop = 0;

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    
        print("Model  " + path + "  Load Complete")
    else:
        print('no trained model! Train model first')
        break;


    if path == "model_dark_rlb511":
        modified = sess.run(decoder,
                   feed_dict={X:dark_data, keeprate: 1})
    elif path =="model_noise_rlb511":
        modified = sess.run(decoder,
                   feed_dict={X:nois_data, keeprate: 1})
    elif path == "model_combine_rlb511":
        modified = sess.run(decoder,
                   feed_dict={X:comb_data, keeprate: 1})

    for i in range(datasize) :
        if i%1000 == 0:
            print(i)

        arr = np.multiply(modified[i,:],255.).astype(np.uint8).reshape(17,17,3);
        img = Image.fromarray(arr, 'RGB')
        if path == "model_dark_rlb511":
            img.save('recon_d/'+str(i)+'.jpg', 'JPEG')
        elif path =="model_noise_rlb511":
            img.save('recon_n/'+str(i)+'.jpg', 'JPEG')
        elif path == "model_combine_rlb511":
            img.save('recon_c/'+str(i)+'.jpg', 'JPEG')
        arr2 = np.multiply(orig_data[i,:],255.).astype(np.uint8).reshape(17,17,3);
        img2 = Image.fromarray(arr2, 'RGB')
        img2.save('recon_o/'+str(i)+'.jpg', 'JPEG')