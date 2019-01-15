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

orig_data=np.empty((datasize,n_input))
dark_data=np.empty((datasize,n_input))
#nois_data=np.empty((datasize,n_input))
#comb_data=np.empty((datasize,n_input))

for i in range (datasize):
    if i%1000 == 0:
        print(i)
    filename_o = 'test_o/' + str(1+i) + '.jpg';
    filename_d = 'test_d/' + str(1+i) + '.jpg';
    #filename_n = 'test_n/' + str(1+i) + '.jpg';
    #filename_c = 'test_c/' + str(1+i) + '.jpg';

    img_o = Image.open( filename_o ).convert('L')
    img_d = Image.open( filename_d ).convert('L')
    #img_n = Image.open( filename_n ).convert('L')
    #img_c = Image.open( filename_c ).convert('L')

    try:
        temp_o = np.asarray( img_o, dtype='uint8' )
    except SystemError:
        temp_o = np.asarray( img_o.getdata(), dtype='uint8' )

    try:
        temp_d = np.asarray( img_d, dtype='uint8' )
    except SystemError:
        temp_d = np.asarray( img_d.getdata(), dtype='uint8' )

#    try:
#        temp_n = np.asarray( img_n, dtype='uint8' )
#    except SystemError:
#        temp_n = np.asarray( img_n.getdata(), dtype='uint8' )

#    try:
#        temp_c = np.asarray( img_c, dtype='uint8' )
#    except SystemError:
#        temp_c = np.asarray( img_c.getdata(), dtype='uint8' )

    temp_o = temp_o.ravel()
    temp_d = temp_d.ravel()
#    temp_n = temp_n.ravel()
#    temp_c = temp_c.ravel()


    orig_data[i,:] = np.true_divide(temp_o,255.);
    dark_data[i,:] = np.true_divide(temp_d,255.);
#   nois_data[i,:] = np.true_divide(temp_n,255.);
#   comb_data[i,:] = np.true_divide(temp_c,255.);


print("Data Load Complete\n")

X = tf.placeholder(tf.float32, [None, n_input])
ORG = tf.placeholder(tf.float32, [None, n_input])
###########Dropout rate set##################
#keeprate = tf.placeholder(tf.float32)
####################################
###########Layer set#########################
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden[0]]))
b_encode = tf.Variable(tf.random_normal([n_hidden[0]]))

encoder = tf.nn.sigmoid(
				tf.add(tf.matmul(X, W_encode), b_encode))
#encoder = tf.nn.dropout(encoder,keeprate);
encoder_pre = tf.nn.sigmoid(
				tf.add(tf.matmul(ORG, W_encode), b_encode))


W_encode1 = tf.Variable(tf.random_normal([n_hidden[0], n_hidden[1]]))
b_encode1 = tf.Variable(tf.random_normal([n_hidden[1]]))


encoder1 = tf.nn.sigmoid(
				tf.add(tf.matmul(encoder, W_encode1), b_encode1))
#encoder1 = tf.nn.dropout(encoder1,keeprate);
encoder1_pre = tf.nn.sigmoid(
				tf.add(tf.matmul(encoder_pre, W_encode1), b_encode1))

W_encode2 = tf.Variable(tf.random_normal([n_hidden[1], n_hidden[2]]))
b_encode2 = tf.Variable(tf.random_normal([n_hidden[2]]))


encoder2 = tf.nn.sigmoid(
				tf.add(tf.matmul(encoder1, W_encode2), b_encode2))
#encoder2 = tf.nn.dropout(encoder2,keeprate);

W_decode2 = tf.Variable(tf.random_normal([n_hidden[2], n_hidden[1]]))
b_decode2 = tf.Variable(tf.random_normal([n_hidden[1]]))

decoder2 = tf.nn.sigmoid(
				tf.add(tf.matmul(encoder2, W_decode2), b_decode2))
#decoder2 = tf.nn.dropout(decoder2,keeprate);

W_decode1 = tf.Variable(tf.random_normal([n_hidden[1], n_hidden[0]]))
b_decode1 = tf.Variable(tf.random_normal([n_hidden[0]]))

decoder1 = tf.nn.sigmoid(
				tf.add(tf.matmul(decoder2, W_decode1), b_decode1))
#decoder1 = tf.nn.dropout(decoder1,keeprate);

decoder1_pre = tf.nn.sigmoid(tf.add(tf.matmul(encoder1,W_decode1),b_decode1))


W_decode = tf.Variable(tf.random_normal([n_hidden[0], n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))

decoder = tf.nn.sigmoid(
				tf.add(tf.matmul(decoder1, W_decode), b_decode))
#decoder = tf.nn.dropout(decoder,keeprate);
decoder_pre = tf.nn.sigmoid(tf.add(tf.matmul(encoder,W_decode),b_decode))

SAVER_DIR = ["model_dark_rlb531_gray"]
			#,"model_noise_rlb531_gray","model_combine_rlb531_gray"]

arr = np.empty((datasize,17,17))
arr2 = np.empty((17,17))

for i in range(datasize) :
    if i%1000 == 0:
        print(i)
    arr2 = np.multiply(orig_data[i,:],255.).reshape(17,17);
    arr2 = arr2.astype(np.uint8)

    img2 = Image.fromarray(arr2,'L')
    img2.save('recon_o/'+str(i)+'.jpg', 'JPEG')

for path in SAVER_DIR:

    saver = tf.train.Saver()
    rgbpath = path;
    ckpt_path = os.path.join(rgbpath,"model")
    ckpt = tf.train.get_checkpoint_state(rgbpath)

    sess = tf.Session()

    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model  " + rgbpath + "  Load Complete")
    else:
        print('no trained model! Train model first')
        break;



    if path[6] == "d":
        modified = sess.run(decoder,
               feed_dict={X:dark_data})
    #elif path[6] =="n":
    #    modified = sess.run(decoder,
    #           feed_dict={X:nois_data})
    #elif path[6] == "c":
    #    modified = sess.run(decoder,
    #           feed_dict={X:comb_data})

    for i in range(datasize) :
        if i%1000 == 0:
            print(i)
        arr = np.multiply(modified[i,:],255.).reshape(17,17);

        img = Image.fromarray(arr.astype(np.uint8), 'L')
        if path[6] == "d":
            img.save('recon_d/'+str(i)+'.jpg', 'JPEG')
        #elif path[6] =="n":
        #    img.save('recon_n/'+str(i)+'.jpg', 'JPEG')
        #elif path[6] == "c":
        #    img.save('recon_c/'+str(i)+'.jpg', 'JPEG')
        
