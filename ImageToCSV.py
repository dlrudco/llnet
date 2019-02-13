import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from pandas import read_csv
from PIL import Image

n_input =   289;#gray :: 289 , rgb :: 867
datasize = 8370;

orig_data=np.empty((datasize,n_input))
dark_data=np.empty((datasize,n_input))
nois_data=np.empty((datasize,n_input))
comb_data=np.empty((datasize,n_input))



for i in range (datasize):
	if i%1000 == 0:
		print(i)
	filename_o = 'test_o/' + str(1+i) + '.jpg';
	filename_d = 'test_d/' + str(1+i) + '.jpg';
	filename_n = 'test_n/' + str(1+i) + '.jpg';
	filename_c = 'test_c/' + str(1+i) + '.jpg';

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
ORIG = pd.DataFrame(orig_data)
DARK = pd.DataFrame(dark_data)
NOIS = pd.DataFrame(nois_data)
COMB = pd.DataFrame(comb_data)

ORIG.to_csv("gray_original_test.csv")
DARK.to_csv("gray_dark_test.csv")
NOIS.to_csv("gray_noise_test.csv")
COMB.to_csv("gray_combine_test.csv")