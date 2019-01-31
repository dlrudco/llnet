import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import read_csv
from sklearn.model_selection import train_test_split
from PIL import Image
from random import randint

def argumentmax(data):
	temp = np.zeros(data.shape)
	for i in range(data.shape[0]):
		index = np.argmax(data[i,:])
		temp[i,index] = 1
	return temp
def loadcsv(data):
	raw_data = read_csv(data)
	return raw_data.values[:,1:]
#########
#Data import
######
n_hidden = [2000, 1600, 1200]; 


n_input =   289;
datasize = 356000;

orig_data=np.empty((datasize,n_input))
dark_data=np.empty((datasize,n_input))
nois_data=np.empty((datasize,n_input))
comb_data=np.empty((datasize,n_input))


try:
	orig_data = loadcsv("gray_original.csv")
	print("orig_data load complete")
	#dark_data = loadcsv("gray_dark.csv")
	#print("dark_data load complete")
	nois_data = loadcsv("gray_noise.csv")
	print("nois_data load complete")
	# comb_data = loadcsv("gray_combine.csv")
	# print("comb_data load complete")
except :
	for i in range (datasize):
		if i%1000 == 0:
			print(i)
		filename_o = 'original/' + str(201+i) + '.jpg';
		filename_d = 'darken/' + str(201+i) + '.jpg';
		filename_n = 'noise/' + str(201+i) + '.jpg';
		filename_c = 'combine/' + str(201+i) + '.jpg';

		img_o = Image.open( filename_o ).convert('L')
		img_d = Image.open( filename_d ).convert('L')
		img_n = Image.open( filename_n ).convert('L')
		img_c = Image.open( filename_c ).convert('L')

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
		#
		try:
			temp_c = np.asarray( img_c, dtype='uint8' )
		except SystemError:
			temp_c = np.asarray( img_c.getdata(), dtype='uint8' )

		temp_o = temp_o[:,:].ravel()
		temp_d = temp_d[:,:].ravel()
		temp_n = temp_n[:,:].ravel()
		temp_c = temp_c[:,:].ravel()


		orig_data[i,:] = np.true_divide(temp_o,255.);
		dark_data[i,:] = np.true_divide(temp_d,255.);
		nois_data[i,:] = np.true_divide(temp_n,255.);
		comb_data[i,:] = np.true_divide(temp_c,255.);


print("Data Load Complete\n")

learning_rate = 0.1
training_epoch = 1380
batch_size = 2000




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
######################################################
#################cost&optimizer set###################
beta = 0.0015;
lamda = 0.000015;

L2norm_total = tf.reduce_mean(tf.square(tf.subtract(ORG,decoder)))
L2norm_pre1 = tf.reduce_mean(tf.square(tf.subtract(ORG,decoder_pre)))
L2norm_pre2 = tf.reduce_mean(tf.square(tf.subtract(encoder,decoder1_pre)))
L2norm_pre12 = tf.reduce_mean(tf.square(tf.subtract(ORG,decoder_pre_l2)))
L2norm_pre3 = tf.reduce_mean(tf.square(tf.subtract(encoder1,decoder2)))

rhohat_e = tf.reduce_mean(encoder,0);
rhohat_e1 = tf.reduce_mean(encoder1,0);
rhohat_e2 = tf.reduce_mean(encoder2,0);

rho_l1 = tf.ones_like(b_encode)* 0.85
rho_l2 = tf.ones_like(b_encode1) * 0.85
rho_l3 = tf.ones_like(b_encode2) * 0.85

#rhohat_d1 = tf.reduce_mean(decoder1_pre,0);
#rhohat_d2 = tf.reduce_mean(decoder2);

log_e = tf.subtract(tf.log(rho_l1),tf.log(tf.clip_by_value(rhohat_e,1e-12,1.)));
log_e_1 = tf.subtract(tf.log(tf.to_float(1)-rho_l1),tf.log(tf.clip_by_value(tf.to_float(1)-rhohat_e,1e-12,1.)));

log_e1 = tf.subtract(tf.log(rho_l2),tf.log(tf.clip_by_value(rhohat_e1,1e-12,1.)));
log_e1_1 = tf.subtract(tf.log(tf.to_float(1)-rho_l2),tf.log(tf.clip_by_value(tf.to_float(1)-rhohat_e1,1e-12,1.)));

log_e2 = tf.subtract(tf.log(rho_l3),tf.log(tf.clip_by_value(rhohat_e2,1e-12,1.)));
log_e2_1 = tf.subtract(tf.log(tf.to_float(1)-rho_l3),tf.log(tf.clip_by_value(tf.to_float(1)-rhohat_e2,1e-12,1.)));

#log_d1 = tf.subtract(tf.log(tf.clip_by_value(rho,1e-8,1.)),tf.log(tf.clip_by_value(rhohat_d1,1e-8,1.)));
#log_d1_1 = tf.subtract(tf.log(tf.clip_by_value(tf.to_float(1)-rho,1e-8,1.)),tf.log(tf.clip_by_value(tf.to_float(1)-rhohat_d1,1e-8,1.)));

#log_d2 = tf.subtract(tf.log(tf.clip_by_value(rho,1e-8,1.)),tf.log(tf.clip_by_value(rhohat_d2,1e-8,1.)));
#log_d2_1 = tf.subtract(tf.log(tf.clip_by_value(tf.to_float(1)-rho,1e-8,1.)),tf.log(tf.clip_by_value(tf.to_float(1)-rhohat_d2,1e-8,1.)));

kl_e = tf.add(tf.multiply(rho_l1,log_e),tf.multiply(tf.subtract(tf.to_float(1),rho_l1),log_e_1))
kl_e1 = tf.add(tf.multiply(rho_l2,log_e1),tf.multiply(tf.subtract(tf.to_float(1),rho_l2),log_e1_1))
kl_e2 = tf.add(tf.multiply(rho_l3,log_e2),tf.multiply(tf.subtract(tf.to_float(1),rho_l3),log_e2_1))
#kl_d1 = tf.add(tf.multiply(rho,log_d1),tf.multiply(tf.subtract(tf.to_float(1),rho),log_d1_1))
#kl_d2 = tf.add(tf.multiply(rho,log_d2),tf.multiply(tf.subtract(tf.to_float(1),rho),log_d2_1))

kl1 = tf.multiply(beta,tf.reduce_mean(kl_e));
kl2 = tf.multiply(beta,tf.reduce_mean(kl_e1));
kl3 = tf.multiply(beta,tf.reduce_mean(kl_e2));


fnorm_e = tf.square(tf.norm(W_encode))/n_hidden[0]
fnorm_e1 = tf.square(tf.norm(W_encode1))/n_hidden[1]
fnorm_e2 = tf.square(tf.norm(W_encode2))/n_hidden[2]

fnorm_d = tf.square(tf.norm(W_decode))/n_hidden[0]
fnorm_d1 = tf.square(tf.norm(W_decode1))/n_hidden[1]
fnorm_d2 = tf.square(tf.norm(W_decode2))/n_hidden[2]

weight_decay_1 = tf.multiply(lamda,tf.add(fnorm_e,fnorm_d));
weight_decay_2 = tf.multiply(lamda,tf.add(fnorm_e1,fnorm_d1));
weight_decay_12 = tf.divide(tf.add(weight_decay_1,weight_decay_1),2.);
weight_decay_3 = tf.multiply(lamda,tf.add(fnorm_e2,fnorm_d2));
weight_decay_tot = tf.add(tf.add(weight_decay_1,weight_decay_2),weight_decay_3)

cost_l1 = tf.add(L2norm_pre1,weight_decay_1)
cost_l2 = tf.add(L2norm_pre2,weight_decay_2)
cost_l1l2 = tf.add(L2norm_pre12,weight_decay_12)
cost_l3 = tf.add(L2norm_pre3,weight_decay_3)

cost_da1 = tf.add(cost_l1,kl1)
cost_da2 = tf.add(cost_l2,kl2)
cost_da3 = tf.add(cost_l3,kl3)


cost_ssda = L2norm_total +  0.3*weight_decay_tot;
#######################################
optimizer_da1 = tf.train.AdamOptimizer(learning_rate).minimize(cost_da1,var_list=[W_encode,W_decode,b_encode,b_decode])
optimizer_da2 = tf.train.AdamOptimizer(learning_rate).minimize(cost_da2,var_list=[W_encode1,W_decode1,b_encode1,b_decode1])
optimizer_da3 = tf.train.AdamOptimizer(0.1*learning_rate).minimize(cost_da3,var_list=[W_encode2,W_decode2,b_encode2,b_decode2])

optimizer_l1_200 = tf.train.AdamOptimizer(learning_rate).minimize(cost_l1,var_list=[W_encode,W_decode,b_encode,b_decode])
optimizer_l1_after = tf.train.AdamOptimizer(0.1*learning_rate).minimize(cost_l1,var_list=[W_encode,W_decode,b_encode,b_decode])

optimizer_l2_200 = tf.train.AdamOptimizer(learning_rate).minimize(cost_l2,var_list=[W_encode1,W_decode1,b_encode1,b_decode1])
optimizer_l2_after = tf.train.AdamOptimizer(0.1*learning_rate).minimize(cost_l2,var_list=[W_encode1,W_decode1,b_encode1,b_decode1])

optimizer_l1l2_200 = tf.train.AdamOptimizer(learning_rate).minimize(cost_l1l2,var_list=[W_encode,W_encode1,W_decode,W_decode1,b_encode,b_decode,b_encode1,b_decode1])
optimizer_l1l2_after = tf.train.AdamOptimizer(0.1*learning_rate).minimize(cost_l1l2,var_list=[W_encode,W_encode1,W_decode,W_decode1,b_encode,b_decode,b_encode1,b_decode1])

optimizer_l3_200 = tf.train.AdamOptimizer(learning_rate).minimize(cost_l3,var_list=[W_encode2,W_decode2,b_encode2,b_decode2])
optimizer_l3_after = tf.train.AdamOptimizer(0.1*learning_rate).minimize(cost_l3,var_list=[W_encode2,W_decode2,b_encode2,b_decode2])

optimizer_ssda_200 = tf.train.AdamOptimizer(learning_rate).minimize(cost_ssda)
optimizer_ssda_after = tf.train.AdamOptimizer(0.1*learning_rate).minimize(cost_ssda)
####################################################

total_batch = int(datasize/batch_size)
SAVER_DIR = ["model_noise_rlb850015000015_gray"]
#,"model_noise_rlb85000200005_gray","model_combine_rlb85000200005_gray"]
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

	saver = tf.train.Saver(max_to_keep=None)
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

		print("Model  " + path + "  Train Start")
		#ckpt_path = os.path.join("model_dark_rlb8500100005_gray_test_l1","model")
		#ckpt = tf.train.get_checkpoint_state("model_dark_rlb8500100005_gray_test_l1")    
		#saver.restore(sess, ckpt.model_checkpoint_path)    
		#print("Model  model_dark_rlb8500100005_gray_test_l1  Load Complete")
		#old = sess.run(decoder_pre,feed_dict={X:orig_data[0:2,:]})
		epoch_set = [60,200]
		epoch = 0;
		flag = 0;
		oldflag = 1;
		#for epoch in range(training_epoch):
		while True:	
			total_cost = 0
			testcost_k = 0
			testcost_l = 0
			testcost_w = 0
			seed=randint(1,20000)
			orig =  np.random.RandomState(seed).permutation(orig_data);
			if path[6] == "d":
				batch = np.random.RandomState(seed).permutation(dark_data)
			elif path[6] =="n":
				batch = np.random.RandomState(seed).permutation(nois_data)
			elif path[6] == "c":
				batch = np.random.RandomState(seed).permutation(comb_data);
			else:
				print("no matching model name!")
				break;
			#test = sess.run(decoder_pre,feed_dict={X:orig_data[0:2,:]})
			
			#if np.all(test == old): 
			#	print("preserved")
			#else:
			#	print("changed what shouldn't change")
			if oldflag != flag:
				ckpt_path = os.path.join(path+"_test_l"+str(flag+1),"model")
				ckpt = tf.train.get_checkpoint_state(path+"_test_l"+str(flag+1))    
				if ckpt and ckpt.model_checkpoint_path:
					saver.restore(sess, ckpt.model_checkpoint_path)    
					print("Model  " + path+"_test_l"+str(flag+1) + "  Load Complete")
					best_cost = np.inf
				else :
					print("Model  " + path+"_test_l"+str(flag+1) + "  Train Start")
					best_cost = np.inf
			oldflag = flag
			if epoch < 60:    
				#print("skip")
				if flag == 0:
					for i in range(total_batch):
						_, cost_val1 = sess.run([optimizer_da1, cost_da1],
							feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
					
						total_cost = total_cost + cost_val1 
						testk1 = sess.run(kl1,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
						testw1 = sess.run(weight_decay_1,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
						testl1 = sess.run(L2norm_pre1,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
						testcost_k = testcost_k+testk1
						testcost_w = testcost_w+testw1
						testcost_l = testcost_l+testl1
					print("Pretraining l1 Epoch:", "%04d" % (epoch + 1),
						"	Avg. cost =", "{:.12f}".format((total_cost)))
					print("testcost k:", "{:.12f}".format(testcost_k))
					print("testcost w:", "{:.12f}".format(testcost_w))
					print("testcost l:", "{:.12f}".format(testcost_l))
				
				# elif flag == 1:
				# 	for i in range(total_batch):
				# 		_, cost_val2 = sess.run([optimizer_da2, cost_da2],
				# 			feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		total_cost = total_cost + cost_val2
				# 		testk2 = sess.run(kl2,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testw2 = sess.run(weight_decay_2,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testl2 = sess.run(L2norm_pre2,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testcost_k = testcost_k+testk2
				# 		testcost_w = testcost_w+testw2
				# 		testcost_l = testcost_l+testl2
				# 		#print(cost_val1.shape)
				# 		#print(cost_val2.shape)
				# 		#print(cost_val3.shape)
				# 	print("Pretraining l2 Epoch:", "%04d" % (epoch),
				# 			"Avg. cost =", "{:.12f}".format((total_cost)))
				# 	print("testcost k:", "{:.12f}".format(testcost_k))
				# 	print("testcost w:", "{:.12f}".format(testcost_w))
				# 	print("testcost l:", "{:.12f}".format(testcost_l))
				# elif flag == 3:
				# 	for i in range(total_batch):
				# 		_, cost_val3 = sess.run([optimizer_da3, cost_da3],
				# 			feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		total_cost = total_cost + cost_val3
				# 		testk3 = sess.run(kl3,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testw3 = sess.run(weight_decay_3,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testl3 = sess.run(L2norm_pre3,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testcost_k = testcost_k+testk3
				# 		testcost_w = testcost_w+testw3
				# 		testcost_l = testcost_l+testl3
				# 		#print(cost_val1.shape)
				# 		#print(cost_val2.shape)
				# 		#print(cost_val3.shape)
				# 	print("Pretraining l3 Epoch:", "%04d" % (epoch),
				# 			"Avg. cost =", "{:.12f}".format((total_cost)))
				# 	print("testcost k:", "{:.12f}".format(testcost_k))
				# 	print("testcost w:", "{:.12f}".format(testcost_w))
				# 	print("testcost l:", "{:.12f}".format(testcost_l))
				# elif flag == 2 or flag == 4:
				# 	epoch = 60 
				else:
					print("invalid flag number flag: "+str(flag))
					break;
			elif epoch < 260:
				if flag == 0:
					for i in range(total_batch):
						_, cost_val = sess.run([optimizer_l1_200, cost_l1],
							feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
						total_cost += cost_val

						testw = sess.run(weight_decay_1,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
						testl = sess.run(L2norm_pre1,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})

						testcost_w = testcost_w+testw
						testcost_l = testcost_l+testl
					print("Finetuning l1 Stage 1 Epoch:", "%04d" % (epoch -60),
						"Avg. cost =", "{:.12f}".format((total_cost)))
					print("testcost w:", "{:.12f}".format(testcost_w))
					print("testcost l:", "{:.12f}".format(testcost_l))
					#flag += 1
					#epoch = -1
					#continue
				
				# elif flag == 1:
				# 	for i in range(total_batch):
				# 		_, cost_val = sess.run([optimizer_l2_200, cost_l2],
				# 			feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		total_cost += cost_val
				# 		testw = sess.run(weight_decay_2,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testl = sess.run(L2norm_pre2,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testcost_w = testcost_w+testw
				# 		testcost_l = testcost_l+testl
				# 	print("Finetuning l2 Stage 1 Epoch:", "%04d" % (epoch -60),
				# 		"Avg. cost =", "{:.12f}".format((total_cost)))
				# 	print("testcost w:", "{:.12f}".format(testcost_w))
				# 	print("testcost l:", "{:.12f}".format(testcost_l))
				# 	#flag += 1
				# 	#epoch = -1
				# 	#continue
				# elif flag == 2:
				# 	for i in range(total_batch):
				# 		_, cost_val = sess.run([optimizer_l1l2_200, cost_l1l2],
				# 			feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		total_cost += cost_val

				# 		testw = sess.run(weight_decay_12,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testl = sess.run(L2norm_pre12,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})

				# 		testcost_w = testcost_w+testw
				# 		testcost_l = testcost_l+testl
				# 	print("Finetuning l1l2 Stage 1 Epoch:", "%04d" % (epoch -60),
				# 		"Avg. cost =", "{:.12f}".format((total_cost)))
				# 	print("testcost w:", "{:.12f}".format(testcost_w))
				# 	print("testcost l:", "{:.12f}".format(testcost_l))
				# elif flag == 3:
				# 	for i in range(total_batch):
				# 		_, cost_val = sess.run([optimizer_l3_200, cost_l3],
				# 			feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		total_cost += cost_val

				# 		testw = sess.run(weight_decay_3,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testl = sess.run(L2norm_pre3,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})

				# 		testcost_w = testcost_w+testw
				# 		testcost_l = testcost_l+testl
				# 	print("Finetuning l3 Stage 1 Epoch:", "%04d" % (epoch -60),
				# 		"Avg. cost =", "{:.12f}".format((total_cost)))
				# 	print("testcost w:", "{:.12f}".format(testcost_w))
				# 	print("testcost l:", "{:.12f}".format(testcost_l))
				# 	#flag += 1
				# 	#epoch = -1
				# 	#continue
				# elif flag == 4:
				# 	for i in range(total_batch):
				# 		_, cost_val = sess.run([optimizer_ssda_200, cost_ssda],
				# 			feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		total_cost += cost_val

				# 		testw = sess.run(weight_decay_tot,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testl = sess.run(L2norm_total,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})

				# 		testcost_w = testcost_w+testw
				# 		testcost_l = testcost_l+testl
				# 	print("Finetuning ssda Stage 1 Epoch:", "%04d" % (epoch -60),
				# 		"Avg. cost =", "{:.12f}".format((total_cost)))
				# 	print("testcost w:", "{:.12f}".format(testcost_w))
				# 	print("testcost l:", "{:.12f}".format(testcost_l))
				
				else:
					print("invalid flag number flag: "+str(flag))
					break;

			else:
				if flag == 0:	
					for i in range(total_batch):
						_, cost_val = sess.run([optimizer_l1_after, cost_l1],
							feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
						total_cost += cost_val
						testw = sess.run(weight_decay_1,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
						testl = sess.run(L2norm_pre1,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})

						testcost_w = testcost_w+testw
						testcost_l = testcost_l+testl
					print("Finetuning l1 Stage 2 Epoch:", "%04d" % (epoch -260 + 1),
							"Avg. cost =", "{:.12f}".format((total_cost)))
					print("testcost w:", "{:.12f}".format(testcost_w))
					print("testcost l:", "{:.12f}".format(testcost_l))
				
				# elif flag == 1:
				# 	for i in range(total_batch):
				# 		_, cost_val = sess.run([optimizer_l2_after, cost_l2],
				# 			feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		total_cost += cost_val
				# 		testw = sess.run(weight_decay_2,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testl = sess.run(L2norm_pre2,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})

				# 		testcost_w = testcost_w+testw
				# 		testcost_l = testcost_l+testl
				# 	print("Finetuning l2 Stage 2 Epoch:", "%04d" % (epoch -260 + 1),
				# 			"Avg. cost =", "{:.12f}".format((total_cost)))
				# 	print("testcost w:", "{:.12f}".format(testcost_w))
				# 	print("testcost l:", "{:.12f}".format(testcost_l))
				# elif flag == 2:
				# 	for i in range(total_batch):
				# 		_, cost_val = sess.run([optimizer_l1l2_after, cost_l1l2],
				# 			feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		total_cost += cost_val
				# 		testw = sess.run(weight_decay_12,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testl = sess.run(L2norm_pre12,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})

				# 		testcost_w = testcost_w+testw
				# 		testcost_l = testcost_l+testl
				# 	print("Finetuning l1l2 Stage 2 Epoch:", "%04d" % (epoch -260 + 1),
				# 			"Avg. cost =", "{:.12f}".format((total_cost)))
				# 	print("testcost w:", "{:.12f}".format(testcost_w))
				# 	print("testcost l:", "{:.12f}".format(testcost_l))
				# elif flag == 3:
				# 	for i in range(total_batch):
				# 		_, cost_val = sess.run([optimizer_l3_after, cost_l3],
				# 			feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		total_cost += cost_val
				# 		testw = sess.run(weight_decay_3,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testl = sess.run(L2norm_pre3,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})

				# 		testcost_w = testcost_w+testw
				# 		testcost_l = testcost_l+testl
				# 	print("Finetuning l3 Stage 2 Epoch:", "%04d" % (epoch -260 + 1),
				# 			"Avg. cost =", "{:.12f}".format((total_cost)))
				# 	print("testcost w:", "{:.12f}".format(testcost_w))
				# 	print("testcost l:", "{:.12f}".format(testcost_l))
				# elif flag ==4:
				# 	for i in range(total_batch):
				# 		_, cost_val = sess.run([optimizer_ssda_after, cost_ssda],
				# 			feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		total_cost += cost_val
				# 		testw = sess.run(weight_decay_tot,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})
				# 		testl = sess.run(L2norm_total,feed_dict={X: batch[i : i + batch_size,:],ORG: orig[i : i + batch_size,:]})

				# 		testcost_w = testcost_w+testw
				# 		testcost_l = testcost_l+testl
				# 	print("Finetuning ssda Stage 2 Epoch:", "%04d" % (epoch -260 + 1),
				# 			"Avg. cost =", "{:.12f}".format((total_cost)))
				# 	print("testcost w:", "{:.12f}".format(testcost_w))
				# 	print("testcost l:", "{:.12f}".format(testcost_l))
				
				else:
					print("invalid flag number flag: "+str(flag))
					break

				if best_cost == np.inf:
					best_cost = total_cost
				elif total_cost == np.nan:
					print("NaN cost!! check parameters")
					break
				elif best_cost>total_cost :
					ckpt_path = os.path.join(path+"_test_l"+str(flag+1),"model")
					ckpt = tf.train.get_checkpoint_state(path+"_test_l"+str(flag+1))
					saver.save(sess, ckpt_path, global_step=training_epoch)
					best_cost = total_cost
					earlystop = 0
					print("model  " + path+"_test_l"+str(flag+1) + "  saved")
				elif best_cost< total_cost:
					earlystop += 1
					if earlystop >30:
						flag += 1
						epoch = -1
						earlystop =0
						if flag > 1:
							ckpt_path = os.path.join(path+"_test_l"+str(flag),"model")
							ckpt = tf.train.get_checkpoint_state(path+"_test_l"+str(flag))
							saver.restore(sess, ckpt.model_checkpoint_path)
							ckpt_path = os.path.join(path,"model")
							ckpt = tf.train.get_checkpoint_state(path)
							saver.save(sess, ckpt_path, global_step=training_epoch)
							print("Final model saved")
							break
					#continue
				else:
					print("Unknown Situation")
					break
			epoch += 1

	print("Optimization for model " + path + " complete")
	#saver.save(sess, ckpt_path, global_step=training_epoch)


	orig_test = loadcsv("gray_original_test.csv")
	print("orig_test load complete")
	#test = loadcsv("gray_dark_test.csv")
	#print("dark_data load complete")
	test = loadcsv("gray_noise_test.csv")
	print("nois_data load complete")
	# test = loadcsv("gray_combine_test.csv")
	# print("comb_data load complete")


	modified = sess.run(decoder_pre,
			   feed_dict={X:test})
	mod_l1 = sess.run(decoder_pre,
				feed_dict={X:orig_test})
	weight = np.zeros((200,419));
	nimg = np.zeros((200,419));
	oimg = np.zeros((200,419));
	temp = np.ones((17,17))
	for i in range(8370) :   
		arr = np.multiply(modified[i,:],255.).reshape(17,17);
		oarr = np.multiply(mod_l1[i,:],255.).reshape(17,17)
		if i%1000 == 0:
			print(i)
			print(arr.astype(np.uint8))
		else: None
		row = i / 135
		col = i % 135
		weight[3*row:3*row+17,3*col:3*col+17] += temp;
		nimg[3*row:3*row+17,3*col:3*col+17] += arr;
		oimg[3*row:3*row+17,3*col:3*col+17] += oarr;

	nimg = np.divide(nimg,weight).astype(np.uint8);
	result = Image.fromarray(nimg.astype(np.uint8), 'L')
	oimg = np.divide(oimg,weight).astype(np.uint8);
	oresult = Image.fromarray(oimg.astype(np.uint8), 'L')
	oresult.show("original")
	if path[6] == "d" :
		result.save('recon_dark_car.jpg', 'JPEG')
	elif path[6] == "n" :
		result.save('recon_noise_car.jpg', 'JPEG')
	elif path[6] == "c" :
		result.save('recon_combine_car.jpg', 'JPEG')
	result.show("reconstructed")
