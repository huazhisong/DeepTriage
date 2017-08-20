from sklearn.preprocessing import LabelBinarizer
from tensorflow.contrib import learn
import pandas as pd 
import tensorflow  as tf
import numpy as np


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

eclipse_data = pd.read_csv('./sort-text-id.csv', encoding='latin-1')
x = eclipse_data.text 
y = eclipse_data.fixer

dev_sample_percentage = 0.1
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
print(dev_sample_index)
x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]

# 处理training data
# document length取90%的分位数
document_length_df = pd.DataFrame([len(xx.split(" ")) for xx in x_train])
document_length = np.int64(document_length_df.quantile(0.8))
vocabulary_processor = learn.preprocessing.VocabularyProcessor(document_length)
x_train = np.array(list(vocabulary_processor.fit_transform(x_train)))
x_dev = np.array(list(vocabulary_processor.transform(x_dev)))

# 处理label
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_dev = lb.transform(y_dev)

# 将训练数据集转换成TFRecord
size = len(y_train)
filename = './train.tfrecords'
writer = tf.python_io.TFRecordWriter(filename)
print('\n training data transform starting >>>>')
for i in range(size):
	y = y_train[i].tostring()
	x = x_train[i].tostring()
	example = tf.train.Example(features=tf.train.Features(feature={
		'label': bytes_feature(y),
		'data': bytes_feature(x)
		}))
	writer.write(example.SerializeToString())

writer.close()
print('\n Training data transformed end<<<')

size = len(y_dev)
filename = './test.tfrecords'
writer = tf.python_io.TFRecordWriter(filename)
print('\n testing data transform starting >>>>')
for i in range(size):
	y = y_dev[i].tostring()
	x = x_dev[i].tostring()
	example = tf.train.Example(features=tf.train.Features(feature={
		'label': bytes_feature(y),
		'data': bytes_feature(x)
		}))
	writer.write(example.SerializeToString())

writer.close()
print('\n testing data transformed end<<<')
