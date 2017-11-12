from __future__ import print_function


import tensorflow as tf
import sys
import time

import pandas as pd
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.datasets import fetch_20newsgroups

parameter_servers = ["10.24.1.32:2225"]
workers = ["10.24.1.201:2225"]
no_of_workers = 1

f = open("new_data_train",'r')
train_target = list()
train_data = list()
for i in f.readlines():
        temp = i.split(' ')
        train_target.append(int(temp[0]))
        train_data.append(' '.join(temp[1:]))
train_target = np.array(train_target)
#print(train_target)
cat = len(set(train_target)) + 1

f = open("new_data_test",'r')
test_target = list()
test_data = list()
for i in f.readlines():
        temp = i.split(' ')
        test_target.append(int(temp[0]))
        test_data.append(' '.join(temp[1:]))
test_target = np.array(test_target)
#print(test_data[0])


vocab = Counter()

#categories = ["comp.graphics","sci.space"]
#newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
#newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

print('total texts in train:',len(train_data))
print('total texts in test:',len(test_data))


vocab = Counter()

for text in train_data:
    for word in text.split(' '):
        vocab[word.lower()]+=1
        
for text in test_data:
    for word in text.split(' '):
        vocab[word.lower()]+=1

total_words = len(vocab)

def get_word_2_index(vocab):
    word2index = {}
    for i,word in enumerate(vocab):
        word2index[word.lower()] = i
        
    return word2index

word2index = get_word_2_index(vocab)

#print("Index of the word 'the':",word2index['the'])


def get_batch(df,i,batch_size,index):
    batches = []
    results = []
    temp_data = []
    temp_target = []
    size = int(len(train_data) / no_of_workers)
    if df == 1:
        temp_data = train_data[index * size : (index + 1) * size]
        temp_target = train_target[index * size : (index + 1) * size]
    else:
        temp_data = test_data
        temp_target = test_target
    texts = temp_data[i*batch_size:i*batch_size+batch_size]
    categories = temp_target[i*batch_size:i*batch_size+batch_size]
    for text in texts:
        layer = np.zeros(total_words,dtype=float)
        for word in text.split(' '):
            layer[word2index[word.lower()]] += 1
            
        batches.append(layer)
        
    for category in categories:
        y = np.zeros((cat),dtype=float)
        y[category] = 1.
        
        results.append(y)
            
     
    return np.array(batches),np.array(results)



# cluster specification

cluster = tf.train.ClusterSpec({"ps":parameter_servers, "worker":workers})

# input flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# start a server for a specific task
server = tf.train.Server(cluster,
                          job_name=FLAGS.job_name,
                          task_index=FLAGS.task_index)

# config
batch_size = 100
learning_rate = 0.5
training_epochs = 10
logs_path = "/tmp/mnist1/1"

n_input = total_words # Words in vocab
n_classes = cat      # Categories: graphics, sci.space and baseball

if FLAGS.job_name == "ps":
  server.join()
elif FLAGS.job_name == "worker":

  # Between-graph replication
  with tf.device(tf.train.replica_device_setter(
    worker_device="/job:worker/task:%d" % FLAGS.task_index,
    cluster=cluster)):

    # count the number of updates
    global_step = tf.get_variable('global_step', [],
                                initializer = tf.constant_initializer(0),
                                trainable = False)

    # input images
    with tf.name_scope('input'):
      # None -> batch size can be any size, 784 -> flattened mnist image
      x = tf.placeholder(tf.float32, shape=[None, n_input], name="x-input")
      # target 10 output classes
      y_ = tf.placeholder(tf.float32, shape=[None, n_classes], name="y-input")

    # model parameters will change during training so we use tf.Variable
    tf.set_random_seed(1)
    with tf.name_scope("weights"):
      W1 = tf.Variable(tf.random_normal([n_input, n_classes]))

    # bias
    with tf.name_scope("biases"):
      b1 = tf.Variable(tf.zeros([n_classes]))

    # implement model
    with tf.name_scope("softmax"):
      # y is our prediction
      out = tf.add(tf.matmul(x,W1),b1)
      #out = tf.nn.sigmoid(out)
      y = tf.nn.softmax(out)

    # specify cost function
    with tf.name_scope('cross_entropy'):
      # this is our cost
      cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))


    # specify optimizer
    with tf.name_scope('train'):
      # optimizer is an "operation" which we can execute in a session
      grad_op = tf.train.GradientDescentOptimizer(learning_rate)
      '''
      rep_op = tf.train.SyncReplicasOptimizer(grad_op,
                                          replicas_to_aggregate=len(workers), 
                                          total_num_replicas=len(workers),
                                          use_locking=True
                                          )
      train_op = rep_op.minimize(cross_entropy, global_step=global_step)
      '''
      train_op = grad_op.minimize(cross_entropy, global_step=global_step)

    '''
    init_token_op = rep_op.get_init_tokens_op()
    chief_queue_runner = rep_op.get_chief_queue_runner()
    '''

    with tf.name_scope('Accuracy'):
      # accuracy
      correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create a summary for our cost and accuracy
    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)

    # merge all summaries into a single "operation" which we can
#execute in a session
    summary_op = tf.summary.merge_all()
    init_op = tf.initialize_all_variables()
    print("Variables initialized ...")

  sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            global_step=global_step,
                            init_op=init_op)

  begin_time = time.time()
  frequency = 100
  with sv.prepare_or_wait_for_session(server.target) as sess:
    '''
    # is chief
    if FLAGS.task_index == 0:
      sv.start_queue_runners(sess, [chief_queue_runner])
      sess.run(init_token_op)
    '''
    # create log writer object (this will log on every machine)
    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # perform training cycles
    start_time = time.time()
    for epoch in range(training_epochs):

      # number of batches in one epoch
      batch_count = int(len(train_data)/(no_of_workers * batch_size))
      print(batch_count)

      count = 0
      for i in range(batch_count):
        batch_x, batch_y =  get_batch(1,i,batch_size,FLAGS.task_index)
        #print(batch_x.shape, batch_y.shape)
        # perform the operations we defined earlier on batch
        _, cost, summary, step = sess.run(
                        [train_op, cross_entropy, summary_op, global_step],
                        feed_dict={x: batch_x, y_: batch_y})
        writer.add_summary(summary, step)

        count += 1
        if count % frequency == 0 or i+1 == batch_count:
          elapsed_time = time.time() - start_time
          start_time = time.time()
          print("Step: %d," % (step+1),
                " Epoch: %2d," % (epoch+1),
                " Batch: %3d of %3d," % (i+1, batch_count),
                " Cost: %.4f," % cost,
                " AvgTime: %3.2fms" % float(elapsed_time*1000/frequency))
          count = 0
    batch_x, batch_y =  get_batch(2,0,len(test_data)-1,FLAGS.task_index)
    #print(batch_x.shape,batch_y.shape)
    print("Test-Accuracy: %2.2f" % sess.run(accuracy, feed_dict={x: batch_x, y_: batch_y}))
    print("Total Time: %3.2fs" % float(time.time() - begin_time))
    print("Final Cost: %.4f" % cost)

  sv.stop()
  print("done")

