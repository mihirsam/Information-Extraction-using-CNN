from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import csv
import tensorflow as tf
from tensorflow import keras

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

#(Key, Value) pairs Dictionary

#Store the keys to train the model
data1  =pd.read_csv('temp1.csv',skiprows=4,nrows=22)
data1.to_csv('dataset1.csv',columns=['Answers','CSD Score'],index=False,header=False)
data2  =pd.read_csv('temp1.csv',skiprows=4)
data2.to_csv('dataset2.csv',columns=['Answers','CSD Score'],index=False,header=False)
data2  =pd.read_csv('dataset2.csv',skiprows=22)
data2.to_csv('dataset3.csv',index=False,header=False)


with open('dataset1.csv', mode='r') as infile:
   		reader = csv.reader(infile)
		model_answers_score= dict((rows[0],rows[1]) for rows in reader)
with open('dataset3.csv', mode='r') as infile:
   		reader = csv.reader(infile)
		sample_answers_score= dict((rows[0],rows[1]) for rows in reader)

print(model_answers_score.keys())
print("Training set len: ", len(model_answers_score))
print("Test set length: ",len(sample_answers_score))

#Store the keys to train the model
data = list(model_answers_score.keys())
train_labels = [4, 2.5, 5, 3, 5, 4.5 ,4.5, 4.5, 4, 4.5, 2, 3, 3, 5, 5, 3.5, 4.5, 4.5, 3.5, 5, 4, 4.5]
#for i in range(0,len(model_answers_score)):
#	train_labels.append(np.array(model_answers_score.values()[i]))

#print(type(train_labels))
#train_labels = list(np.array(model_answers_score.values()))

testdata = list(sample_answers_score.keys())
test_labels = [4.5 , 4, 3.5 , 5, 4.5, 4.5, 5, 4.5]
#test_labels = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]
#for i in range(0,len(sample_answers_score)):
#	test_labels.append(np.array(sample_answers_score.values()[i]))

#from gensim.models.doc2vec import Doc2Vec
#from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#from nltk.tokenize import word_tokenize

#Load the model
model= Doc2Vec.load("d2v.model")
#docvec = model.docvecs[1]
#print np.array(docvec)
train_data=[]
for i in range(0,len(model_answers_score)):
	docvec =  model.docvecs[i]
	train_data.append(np.array(docvec))
print(np.array(train_data))
print(len(train_data))
#training_labels=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
#train_labels=np.asarray(training_labels)



#Load the model
model_test = Doc2Vec.load("d2v_test.model")
test_data = []
for i in range(0,len(sample_answers_score)):
	docvec =  model_test.docvecs[i]
	test_data.append(np.array(docvec))
print(np.array(test_data))
print(len(test_data))


'''print("\n")
print(np.array(train_labels))
print("\n")
print(np.array(test_labels))'''

training_data=np.asarray(train_data)
training_labels=np.asarray(train_labels)
#print(type(training_labels))

testing_data=np.asarray(test_data)
testing_labels=np.asarray(test_labels)
#print(type(testing_labels))

#BUILDING MODEL AND TESTING ACCURACY
model_a = keras.Sequential([
    keras.layers.Dense(50, input_shape=(50,), activation=tf.nn.relu),
    keras.layers.Dense(64),
    keras.layers.Dense(48),
    keras.layers.Dense(11, activation=tf.nn.softmax)
])

model_a.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_a.fit(training_data, training_labels, epochs=800)
#model_a.summary()
list1=[]
list1=model_a.predict(testing_data)
for i in range(len(list1)):
	s=0
	pos=0
	for j in range(len(list1[i])):
		if list1[i][j]>s:
			s=list1[i][j]
#			print(list[i][j])
			pos=j
	print(i,":",pos*.5)
test_loss, test_acc = model_a.evaluate(testing_data, testing_labels)

print('Test accuracy:', test_acc)

