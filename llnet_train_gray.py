import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import read_csv
from sklearn.model_selection import train_test_split
from PIL import Image

def argumentmax(data):
	temp = np.zeros(data.shape)
	for i in range(data.shape[0]):
		index = np.argmax(data[i,:])
		temp[i,index] = 1
	return temp

#########
#Data import
######
n_hidden = [2000, 1600, 1200]; 


n_input =   289;
datasize = 356000;

orig_data=np.empty((datasize,n_input))
dark_data=np.empty((datasize,n_input))
#nois_data=np.empty((datasize,n_input))
#comb_data=np.empty((datasize,n_input))

for i in range (datasize):
	if i%1000 == 0:
		print(i)
	filename_o = 'original/' + str(201+i) + '.jpg';
	filename_d = 'darken/' + str(201+i) + '.jpg';
	#filename_n = 'noise/' + str(201+i) + '.jpg';
	#filename_c = 'combine/' + str(201+i) + '.jpg';

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

#	try:
#		temp_n = np.asarray( img_n, dtype='uint8' )
#	except SystemError:
#		temp_n = np.asarray( img_n.getdata(), dtype='uint8' )
#
#	try:
#		temp_c = np.asarray( img_c, dtype='uint8' )
#	except SystemError:
#		temp_c = np.asarray( img_c.getdata(), dtype='uint8' )

	temp_o = temp_o[:,:].ravel()
	temp_d = temp_d[:,:].ravel()
#	temp_n = temp_n[:,:].ravel()
#	temp_c = temp_c[:,:].ravel()


	orig_data[i,:] = np.true_divide(temp_o,255.);
	dark_data[i,:] = np.true_divide(temp_d,255.);
#	nois_data[i,:] = np.true_divide(temp_n,255.);
#	comb_data[i,:] = np.true_divide(temp_c,255.);


print("Data Load Complete\n")

learning_rate = 0.1
training_epoch = 1290
batch_size = 2000




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

#W_decode2 = tf.Variable(tf.random_normal([n_hidden[2], n_hidden[1]]))
W_decode2 = tf.transpose(W_encode2)
b_decode2 = tf.Variable(tf.random_normal([n_hidden[1]]))

decoder2 = tf.nn.sigmoid(
				tf.add(tf.matmul(encoder2, W_decode2), b_decode2))
#decoder2 = tf.nn.dropout(decoder2,keeprate);

#W_decode1 = tf.Variable(tf.random_normal([n_hidden[1], n_hidden[0]]))
W_decode1 = tf.transpose(W_encode1)
b_decode1 = tf.Variable(tf.random_normal([n_hidden[0]]))

decoder1 = tf.nn.sigmoid(
				tf.add(tf.matmul(decoder2, W_decode1), b_decode1))
#decoder1 = tf.nn.dropout(decoder1,keeprate);

decoder1_pre = tf.nn.sigmoid(tf.add(tf.matmul(encoder1,W_decode1),b_decode1))


#W_decode = tf.Variable(tf.random_normal([n_hidden[0], n_input]))
W_decode = tf.transpose(W_encode)
b_decode = tf.Variable(tf.random_normal([n_input]))

decoder = tf.nn.sigmoid(
				tf.add(tf.matmul(decoder1, W_decode), b_decode))
#decoder = tf.nn.dropout(decoder,keeprate);
decoder_pre = tf.nn.sigmoid(tf.add(tf.matmul(encoder,W_decode),b_decode))
######################################################
#################cost&optimizer set###################
rho = 0.0001;
beta = 1.;
lamda = 1.;

L2norm_total = tf.divide(tf.reduce_mean(tf.square(tf.subtract(ORG,decoder))),(2*batch_size));
L2norm_pre1 = tf.divide(tf.reduce_mean(tf.square(tf.subtract(ORG,decoder_pre))),(2*batch_size));
L2norm_pre2 = tf.divide(tf.reduce_mean(tf.square(tf.subtract(encoder_pre,decoder1_pre))),(2*batch_size));
L2norm_pre3 = tf.divide(tf.reduce_mean(tf.square(tf.subtract(encoder1_pre,decoder2))),(2*batch_size));

rhohat_e = tf.reduce_mean(encoder,0);
rhohat_e1 = tf.reduce_mean(encoder1,0);
rhohat_e2 = tf.reduce_mean(encoder2,0);

#rhohat_d1 = tf.reduce_mean(decoder1_pre,0);
#rhohat_d2 = tf.reduce_mean(decoder2);

log_e = tf.subtract(tf.log(rho),tf.log(tf.clip_by_value(rhohat_e,1e-8,1.)));
log_e_1 = tf.subtract(tf.log(tf.to_float(1)-rho),tf.log(tf.clip_by_value(tf.to_float(1)-rhohat_e,1e-8,1.)));

log_e1 = tf.subtract(tf.log(rho),tf.log(tf.clip_by_value(rhohat_e1,1e-8,1.)));
log_e1_1 = tf.subtract(tf.log(tf.to_float(1)-rho),tf.log(tf.clip_by_value(tf.to_float(1)-rhohat_e1,1e-8,1.)));

log_e2 = tf.subtract(tf.log(rho),tf.log(tf.clip_by_value(rhohat_e2,1e-8,1.)));
log_e2_1 = tf.subtract(tf.log(tf.to_float(1)-rho),tf.log(tf.clip_by_value(tf.to_float(1)-rhohat_e2,1e-8,1.)));

#log_d1 = tf.subtract(tf.log(tf.clip_by_value(rho,1e-8,1.)),tf.log(tf.clip_by_value(rhohat_d1,1e-8,1.)));
#log_d1_1 = tf.subtract(tf.log(tf.clip_by_value(tf.to_float(1)-rho,1e-8,1.)),tf.log(tf.clip_by_value(tf.to_float(1)-rhohat_d1,1e-8,1.)));

#log_d2 = tf.subtract(tf.log(tf.clip_by_value(rho,1e-8,1.)),tf.log(tf.clip_by_value(rhohat_d2,1e-8,1.)));
#log_d2_1 = tf.subtract(tf.log(tf.clip_by_value(tf.to_float(1)-rho,1e-8,1.)),tf.log(tf.clip_by_value(tf.to_float(1)-rhohat_d2,1e-8,1.)));

kl_e = tf.add(tf.multiply(rho,log_e),tf.multiply(tf.subtract(tf.to_float(1),rho),log_e_1))
kl_e1 = tf.add(tf.multiply(rho,log_e1),tf.multiply(tf.subtract(tf.to_float(1),rho),log_e1_1))
kl_e2 = tf.add(tf.multiply(rho,log_e2),tf.multiply(tf.subtract(tf.to_float(1),rho),log_e2_1))
#kl_d1 = tf.add(tf.multiply(rho,log_d1),tf.multiply(tf.subtract(tf.to_float(1),rho),log_d1_1))
#kl_d2 = tf.add(tf.multiply(rho,log_d2),tf.multiply(tf.subtract(tf.to_float(1),rho),log_d2_1))

kl1 = tf.multiply(beta,tf.reduce_mean(kl_e));
kl2 = tf.multiply(beta,tf.reduce_mean(kl_e1));
kl3 = tf.multiply(beta,tf.reduce_mean(kl_e2));


fnorm_e = tf.divide(tf.square(tf.norm(W_encode)),(2*batch_size));
fnorm_e1 = tf.divide(tf.square(tf.norm(W_encode1)),(2*batch_size));
fnorm_e2 = tf.divide(tf.square(tf.norm(W_encode2)),(2*batch_size));

fnorm_d = tf.divide(tf.square(tf.norm(W_decode)),(2*batch_size));
fnorm_d1 = tf.divide(tf.square(tf.norm(W_decode1)),(2*batch_size));
fnorm_d2 = tf.divide(tf.square(tf.norm(W_decode2)),(2*batch_size));

weight_decay_tot = tf.multiply(lamda,tf.add(tf.add(tf.add(tf.add(tf.add(fnorm_e,fnorm_e1),fnorm_e2),fnorm_d1),fnorm_d2),fnorm_d));
weight_decay_1 = tf.multiply(lamda,tf.add(fnorm_e,fnorm_d1));
weight_decay_2 = tf.multiply(lamda,tf.add(fnorm_e1,fnorm_d2));
weight_decay_3 = tf.multiply(lamda,fnorm_e2);

cost_da1 = L2norm_pre1 + weight_decay_1 +kl1
cost_da2 = L2norm_pre2 + weight_decay_2 +kl2
cost_da3 = L2norm_pre3 + weight_decay_3 +kl3

cost_ssda = L2norm_total + weight_decay_tot;
#######################################
optimizer_da1 = tf.train.AdamOptimizer(learning_rate).minimize(cost_da1)
optimizer_da2 = tf.train.AdamOptimizer(learning_rate).minimize(cost_da2)
optimizer_da3 = tf.train.AdamOptimizer(0.1*learning_rate).minimize(cost_da3)

optimizer_ssda_200 = tf.train.AdamOptimizer(learning_rate).minimize(cost_ssda)
optimizer_ssda_after = tf.train.AdamOptimizer(0.1*learning_rate).minimize(cost_ssda)
####################################################

total_batch = int(datasize/batch_size)
SAVER_DIR = ["model_dark_rlb000111_gray"]
			#,"model_noise_rlb531_gray","model_combine_rlb531_gray"]
			#,"model_dark_rlb10101_1","model_noise_rlb10101_1","model_combine_rlb10101_1"
			#,"model_dark_rlb10101_2","model_noise_rlb10101_2","model_combine_rlb10101_2"]

#ind = 0;
for path in SAVER_DIR:
	#channel = int(ind/3) % 3;
	#ind += 1
	#if channel != int(path[-1]):
	#	print("channel assignment error! pls check")
	#	break;

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
	else :
		best_cost = np.inf

		print("Model  " + path + "  Train Start")    

		for epoch in range(training_epoch):
			total_cost = 0
			orig =  np.random.RandomState(epoch).permutation(orig_data);
			if path[6] == "d":
				batch = np.random.RandomState(epoch).permutation(dark_data)
			elif path[6] =="n":
				batch = np.random.RandomState(epoch).permutation(nois_data)
			elif path[6] == "c":
				batch = np.random.RandomState(epoch).permutation(comb_data);
			
			if epoch < 30:    
				for i in range(total_batch):
					_, cost_val1 = sess.run([optimizer_da1, cost_da1],
						feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
					total_cost = total_cost + cost_val1 

				print("Pretraining DA1 Epoch:", "%04d" % (epoch + 1),
						"Avg. cost =", "{:.12f}".format((total_cost)/ (total_batch)))
			elif epoch < 60:    
				for i in range(total_batch):
					_, cost_val2 = sess.run([optimizer_da2, cost_da2],
						feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
					total_cost = total_cost + cost_val2
					#print(cost_val1.shape)
					#print(cost_val2.shape)
					#print(cost_val3.shape)
				print("Pretraining Epoch:", "%04d" % (epoch -29),
						"Avg. cost =", "{:.12f}".format((total_cost)/ (total_batch)))
			elif epoch < 90:    
				for i in range(total_batch):
					_, cost_val3 = sess.run([optimizer_da3, cost_da3],
						feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
					total_cost = total_cost + cost_val3
					#print(cost_val1.shape)
					#print(cost_val2.shape)
					#print(cost_val3.shape)
				print("Pretraining Epoch:", "%04d" % (epoch -59),
						"Avg. cost =", "{:.12f}".format((total_cost)/ (total_batch)))
			elif epoch < 290:
				for i in range(total_batch):
					_, cost_val = sess.run([optimizer_ssda_200, cost_ssda],
						feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
					total_cost += cost_val
				print("Finetuning Stage 1 Epoch:", "%04d" % (epoch -89),
						"Avg. cost =", "{:.12f}".format((total_cost)/ total_batch))
			else:
				for i in range(total_batch):
					_, cost_val = sess.run([optimizer_ssda_after, cost_ssda],
						feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
					total_cost += cost_val
				print("Finetuning Stage 2 Epoch:", "%04d" % (epoch -290 + 1),
						"Avg. cost =", "{:.12f}".format((total_cost)/ total_batch))

				if best_cost == np.inf:
					best_cost = total_cost;
				elif total_cost == np.nan:
					print("NaN cost!! check parameters")
					break
				elif best_cost>total_cost :
					saver.save(sess, ckpt_path, global_step=training_epoch)
					best_cost = total_cost
					print("model saved")
				elif best_cost*1.05 < total_cost:
					break
					#continue

	print("Optimization for model " + path + " complete")
	#saver.save(sess, ckpt_path, global_step=training_epoch)

#Impute = sess.run(decoder,
#                   feed_dict={X:traindata.values, keeprate: 1})

#Impute = np.multiply(traindata.values , trainmissmat) + np.multiply(Impute , trainsw_missmat)

#Impute[:,4:6] = argumentmax(Impute[:,4:6])
#Impute[:,8:10] = argumentmax(Impute[:,8:10])
#Impute[:,24:26] = argumentmax(Impute[:,24:26])
#Impute[:,29:31] = argumentmax(Impute[:,29:31])
#Impute[:,35:38] = argumentmax(Impute[:,35:38])
#Impute[:,41:46] = argumentmax(Impute[:,41:46])
#Impute[:,48:50] = argumentmax(Impute[:,48:50])
#Impute[:,51:54] = argumentmax(Impute[:,51:54])
#Impute[:,54:60] = argumentmax(Impute[:,54:60])
#Impute[:,60:65] = argumentmax(Impute[:,60:65])

#print(Impute - data.values)
#print(data)
#Impute =pd.DataFrame(Impute)
#Impute.to_csv("trainimp_AUTO.csv")
#print("Train Imputation complete")

#Impute = sess.run(decoder,
#                   feed_dict={X:testdata.values, keeprate: 1})

#Impute = np.multiply(testdata.values , testmissmat) + np.multiply(Impute , testsw_missmat)

#Impute[:,4:6] = argumentmax(Impute[:,4:6])
#Impute[:,8:10] = argumentmax(Impute[:,8:10])
#Impute[:,24:26] = argumentmax(Impute[:,24:26])
#Impute[:,29:31] = argumentmax(Impute[:,29:31])
#Impute[:,35:38] = argumentmax(Impute[:,35:38])
#Impute[:,41:46] = argumentmax(Impute[:,41:46])
#Impute[:,48:50] = argumentmax(Impute[:,48:50])
#Impute[:,51:54] = argumentmax(Impute[:,51:54])
#Impute[:,54:60] = argumentmax(Impute[:,54:60])
#Impute[:,60:65] = argumentmax(Impute[:,60:65])

#print(Impute - data.values)
#print(data)
#Impute =pd.DataFrame(Impute)
#Impute.to_csv("testimp_AUTO.csv")
#print("Test Imputation complete")
