# univariate multi-step encoder-decoder lstm
import numpy as np
import pandas as pd
from math import sqrt
from numpy import split
from numpy import array
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
# from keras.models import Sequential
from tensorflow.keras import Sequential
# from keras.layers import Dense
from tensorflow.keras.layers import Dense, Flatten, LSTM, RepeatVector, TimeDistributed
# from keras.layers import Flatten
# from keras.layers import LSTM
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed
import datetime
import tensorflow as tf
import keras
import sys

batch_size=int(sys.argv[1])
epochs=int(sys.argv[2])
neurons=int(sys.argv[3])
ndays_in=int(sys.argv[4])
ndays_out=int(sys.argv[5])
iscale=int(sys.argv[6])
nonlinear=sys.argv[7]
run_label=sys.argv[8]

# this did not work
class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print('Training: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_train_batch_end(self, batch, logs=None):
    print('Training: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_begin(self, batch, logs=None):
    print('Evaluating: batch {} begins at {}'.format(batch, datetime.datetime.now().time()))

  def on_test_batch_end(self, batch, logs=None):
    print('Evaluating: batch {} ends at {}'.format(batch, datetime.datetime.now().time()))
	
# split a univariate dataset into train/test sets
def split_dataset(data,ntrain,ntest):
# see LSTM_encoder_decoder for MachineLearningMastery version
# from which this is derived

#    print(' data.shape ',data.shape)
#    print('data[0]',data[0])
    if len(data) == ntrain+ntest: # just a check
        ii=np.arange(ntrain+ntest)
        np.random.seed() # randomize split each time
        index_train=np.random.choice(ii,size=ntrain,replace=False)
        index_test=np.setdiff1d(ii,index_train)
        train=data[index_train,:]
        test=data[index_test,:]
#        train=data.take(index_train)
#        test=data.take(index_test)
#        train, test = data[0:ntrain], data[ntrain:ntrain+ntest] # these ranges do not overlap because
# ntrain at the top of a range is not at an included point, but ntrain at the bottom of
# a range is. love python, don't we?

#        print(' train.shape ',train.shape,' test.shape ',test.shape)
    else:
        print('ntrain+ntest .ne. len(data)',ntrain,ntest,len(data))
    
    return train, test
 
# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores
 
# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
 
# train the model
def build_model(train_x,train_y, test_x,test_y,timesteps,ndays_in,features,batch_size,epochs,neurons):
#    print('train_x.shape ',train_x.shape,'train_y.shape ',train_y.shape)
#    verbose, epochs, batch_size = 1, 20, 80
    verbose = 0 # , epochs = 1, 20
# reshape train_x and train_y into [samples, timesteps, features] - ***this is trivial***
    train_x=train_x.reshape((train_x.shape[0],ndays_in,features)) # train_x.reshape((train_x.shape[0],train_x.shape[1],1))
    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
    test_x=test_x.reshape((test_x.shape[0],ndays_in,features)) # test_x.reshape((train_x.shape[0],train_x.shape[1],1))
    test_y = test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
#    print('train_y.shape ',train_y.shape,' train_x.shape ',train_x.shape)
# define model
    # neurons=200
    model = Sequential()
    model.add(LSTM(neurons, activation=nonlinear, input_shape=(n_timesteps, n_features)))
    model.add(RepeatVector(n_outputs)) # this makes n_outputs (rather than 1, as in Dense(1))
    model.add(LSTM(neurons, activation=nonlinear, return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    model.add(TimeDistributed(Dense(1)))
    model.compile(loss='mse', optimizer='adam')
# fit network
#    history=model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose, \
#                     callbacks=[MyCustomCallback()])
    history=model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(test_x,test_y))
    return model, history
 
# evaluate a single model
def evaluate_model(train_x,train_y,test_x,test_y,timesteps,ndays_in,features,cmax,dmax,batch_size,epochs,neurons):
# fit model
    model, history = build_model(train_x,train_y,test_x,test_y,timesteps,ndays_in,features,batch_size,epochs,neurons)
    print(' back from build_model ')
    count=test_x.shape[0]
    predictions = list()
# trivial reshape of test_x,y just as done to train_x,y in build_model
    test_x=test_x.reshape((test_x.shape[0],ndays_in,features)) # test_x.reshape((test_x.shape[0],test_x.shape[1],1))
    test_y=test_y.reshape((test_y.shape[0], test_y.shape[1], 1))
    ssq=np.zeros(7)
    deviation=np.zeros((timesteps,count))
    deviationr=np.zeros((timesteps,count))
    date1=np.zeros(count)
#    count=200
    for i in range(count):
# predict the week
        input_x=test_x[i]
        date1[i]=input_x[0,3]
        input_x=input_x.reshape((1,ndays_in,features))
        yhat = model.predict(input_x, verbose=0)
        yhat=yhat[0]
        yhat=yhat[:,0]
        test_y1=test_y[i]
        test_y2=test_y1[:,0]
        diff1=yhat-test_y2
        deviation[:,i]=diff1
        deviationr[:,i]=diff1/(1.0+test_y2)
#        print('diff1 ',diff1,' diff1*diff1 ',diff1*diff1)
        ssq=ssq+diff1*diff1
#        print('yhat ',yhat)
#        print('test_y2 ',test_y2)
#        print('diff1*diff1 ',diff1*diff1)
#        print(' i ',i,ssq)
# store the predictions
#        predictions.append(yhat_sequence)
# get real observation and add to history for predicting the next week
#        history.append(test[i, :])
# evaluate predictions days for each week
    print(' ssq ',ssq)
    ssqtot=np.sqrt(ssq/count)
    print('ssqtot ',ssqtot)
    return ssqtot,deviation,deviationr,model,history,date1

def future7(model, current_x, ndays_in, timesteps, features):
    count=current_x.shape[0]
    print('count current_x',count)
    future=np.zeros((count,ndays_in+timesteps))
    current_x=current_x.reshape((current_x.shape[0],ndays_in,features))
    for i in range (count):
        input_x=current_x[i]
        input_x=input_x.reshape((1,ndays_in,features))
        yhat=model.predict(input_x,verbose=0)
        yhat=yhat[0]
        ihat=input_x[0]
        yhat=yhat[:,0]
        ihat=ihat[:,0]
        yout=np.hstack((ihat,yhat))
        future[i,:]=yout
        
    return future
	
# load the new file
ipoprho=2 # ipoprho=1 use pop data, ipoprho=2 use pop data and dates
if ipoprho == 1:
    dataset = read_csv('covid_data_3col.csv', header=0)
elif ipoprho == 2:
    if ndays_in == 7:
        dataset = read_csv('covid_data_4col_06092020.csv', header=0)
    else:
        dataset = read_csv('covid_data_4col_06292020x.csv',header=0)
else:
    dataset = read_csv('covid_data_2col.csv', header=0)
    
# run_label="run1"
# iscale=5
# ndays_in=7
# ndays_out=7
ndays=ndays_in+ndays_out
L1=3+4*ndays

file1=open('history_loss' + run_label + '.txt','w')
file2=open('ssq' + run_label + '.txt','w')
file3=open('future' + run_label + '.txt','w')
file4=open('history_val' + run_label + '.txt','w')

# split into train and test
print('dataset.shape',dataset.shape)
dtemp=dataset.to_numpy()
dtemp1=dtemp[:,55]
dmax=1
cmax=1
rmax=1
daymax=1
cscale=1
pscale=1
dayscale=1
if iscale == 1:
    dmax=np.amax(dtemp[:,L1-4])

if iscale == 2:
    dmax=np.amax(dtemp[:,L1-4])
    cmax=np.amax(dtemp[:,L1-3])
    rmax=np.amax(dtemp[:,L1-2])
    daymax=np.amax(dtemp[:,L1-1])
    
if iscale == 3:
    cscale=np.amax(dtemp[:,L1-4])/np.amax(dtemp[:,L1-3])
if iscale == 4:
    cscale=np.amax(dtemp[:,L1-4])/np.amax(dtemp[:,L1-3])
    pscale=np.amax(dtemp[:,L1-4])/np.amax(dtemp[:,L1-2])
if iscale == 5:
    cscale=np.amax(dtemp[:,L1-4])/np.amax(dtemp[:,L1-3])
    pscale=np.amax(dtemp[:,L1-4])/np.amax(dtemp[:,L1-2])
    dayscale=np.amax(dtemp[:,L1-4])/np.amax(dtemp[:,L1-1])
        
print('iscale',iscale,'cscale',cscale,'pscale',pscale,'dayscale',dayscale)

ntrain=3*int(dataset.shape[0]/4)
ntest=dataset.shape[0]-ntrain
print('ntrain ',ntrain,'ntest ',ntest)
loopmax=1
for loop1 in range(loopmax):
    file5=open('deviation' + run_label + str(loop1) + '.txt','w')
    train, test = split_dataset(dataset.values,ntrain,ntest) # dataset.values returns numpy array (?)

 # should work with new ndays
    if iscale == 1:
        train[:,3::4]=(train[:,3::4]/dmax)
        test[:,3::4]=(test[:,3::4]/dmax)
        
    if iscale == 2:
        train[:,3::4]=(train[:,3::4]/dmax)
        test[:,3::4]=(test[:,3::4]/dmax)
        train[:,4::4]=(train[:,4::4]/cmax)
        test[:,4::4]=(test[:,4::4]/cmax)
        train[:,5::4]=(train[:,5::4]/rmax)
        test[:,5::4]=(test[:,5::4]/rmax)
        train[:,6::4]=(train[:,6::4]/daymax)
        test[:,6::4]=(test[:,6::4]/daymax)
        
    if iscale == 3:
        train[:,4::4]=cscale*train[:,4::4]
        test[:,4::4]=cscale*test[:,4::4]
        
    if iscale == 4:
        train[:,4::4]=cscale*train[:,4::4]
        test[:,4::4]=cscale*test[:,4::4]
        train[:,5::4]=pscale*train[:,5::4]
        test[:,5::4]=pscale*test[:,5::4]
    
    if iscale == 5:
        train[:,4::4]=cscale*train[:,4::4]
        test[:,4::4]=cscale*test[:,4::4]
        train[:,5::4]=pscale*train[:,5::4]
        test[:,5::4]=pscale*test[:,5::4]
        train[:,6::4]=dayscale*train[:,6::4]
        test[:,6::4]=dayscale*test[:,6::4]
    
# evaluate model and get scores
    n_input = 7
    train=train[:,3:] # works - changed from 2 to 3 with introduction of 
    county_test=test[:,0] # for analysis - not used in calculation
    test=test[:,3:]   # new final data column in covid_data_4colb.csv
    if ipoprho == 1:
        features=3
    elif ipoprho == 2:
        features=4
    else:
        features=2

    timesteps=7
    train_x=train[:,0:features*ndays_in] # previously 7
    train_y=train[:,features*ndays_in::features] # skip over features in target half of data
      
# hold=input('waiting')
    test_x=test[:,0:features*ndays_in]
    test_y=test[:,features*ndays_in::features]
# Build, compile, fit and evaluate model
    ssq_out, deviation, deviationr, model,history,date1 = evaluate_model(train_x,train_y,test_x,test_y,timesteps, \
                                                             ndays_in,features,cmax,dmax,batch_size,epochs,neurons)
    print(' history keys ',history.history.keys())
    print(' history ',history.history, ' loss ',history.history['loss'])
    np.savetxt(file1,np.column_stack(history.history['loss']))
    np.savetxt(file4,np.column_stack(history.history['val_loss']))
    np.savetxt(file2,np.column_stack(ssq_out))
    pyplot.figure(1)
    pyplot.plot(history.history['loss'])
    pyplot.figure(2)
    pyplot.plot(ssq_out)
#    z1=deviation[0,:]
#    density=test_x[:,2]
#    pyplot.figure(2)
#    pyplot.scatter(z1*dmax,(date1+0.5)*daymax,s=4,c=density,cmap='autumn')

    count=test_x.shape[0]
    row1=np.zeros(17)
    for ii in range(count):
        row1[0]=county_test[ii]
        row1[1]=test_x[ii,2] # population density
        row1[2]=test_x[ii,3] # first day
        row1[3::2]=deviation[:,ii]
        row1[4::2]=test_y[ii]
        
        if ii == 0:
            output=row1
        else:
            output=np.vstack((output,row1))
            
    np.savetxt(file5,output,delimiter=',',header="loc,pop,day1,dev1,ty1,dev2,ty2," + \
    "dev3,ty3,dev4,ty4,dev5,ty5,dev6,ty6,dev7,ty7",comments='')

#pyplot.figure(2)
#fig=matplotlib.pyplot.gcf()
#fig.set_size_inches(4.0,4.0)
#pyplot.savefig('dev_vs_date_' + run_label + str(loop1) + '.png')

# start processing future
    dtemp=dataset.to_numpy()
    count=dtemp.shape[0]
    countb=dtemp.sum(axis=0)[2]
    icountb=countb.astype(int)

    current_x=np.zeros((icountb,features*ndays_in))
    countb=-1
    for i in range (count):
        switch=dtemp[i,2].astype(int)
        location=dtemp[i,0].astype(int)
# third column in e.g. covid_data_4col_06092020.csv is a 1 if this is the
# last row for a given location.
        if switch == 1:
            countb+=1
            current_x[countb,:]=dtemp[i,3+timesteps*features:3+(ndays_in+timesteps)*features]
			
# there was confusion here. current_x is the last ndays_in days of data. These are input
# to predict to get the 7 days *after* the last day of data. There are 3+(ndays_in+timesteps)*features
# total columns. starting point is thus that minus ndays_in*features.

    countd=current_x.sum(axis=0)
    if iscale == 1:
        current_x[:,0::4]=current_x[:,0::4]/dmax
        
    if iscale == 3:
        current_x[:,1::4]=cscale*current_x[:,1::4]
        
    if iscale == 4:
        current_x[:,1::4]=cscale*current_x[:,1::4]
        current_x[:,2::4]=pscale*current_x[:,2::4]
        
    if iscale == 5:
        current_x[:,1::4]=cscale*current_x[:,1::4]
        current_x[:,2::4]=pscale*current_x[:,2::4]
        current_x[:,3::4]=dayscale*current_x[:,3::4]
        
    future = future7(model, current_x,ndays_in,timesteps,features)
    
    total=future.sum(axis=0)
    np.savetxt(file3,np.column_stack(total))
    dtotal=np.zeros(ndays_in+timesteps-1)
    dtotal[0:ndays_in+timesteps-2]=total[1:ndays_in+timesteps-1]-total[0:ndays_in+timesteps-2]
    pyplot.figure(3)
    pyplot.plot(total)
    for i in range (ndays_in+timesteps-1):
        print(' i ',i,'dtotal[i]',dtotal[i],'total[i+1]',total[i+1],'total[i]',total[i])
    days=np.arange(ndays_in+timesteps-1) # for a plot we don't do
    file5.close()

file1.close()
file2.close()
file3.close()
file4.close()
#f5 = pyplot.figure()
#pyplot.bar(days,dtotal,align='center',alpha=0.5)
#pyplot.show()
pyplot.figure(1)
pyplot.savefig('history' + run_label + '.png')
pyplot.figure(2)
pyplot.savefig('ssq' + run_label + '.png')
pyplot.figure(3)
pyplot.savefig('future' + run_label + '.png')
