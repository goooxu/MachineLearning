import numpy as np
import tensorflow as tf
from six.moves import cPickle

image_size = 28
num_labels = 10
pickle_file = 'notMNIST.pickle'

model_file = 'models/notMNIST.ckpt'

with open(pickle_file, 'rb') as f:
    obj = cPickle.load(f)

    train_dataset = obj['train_dataset']
    train_labels = obj['train_labels']
    valid_dataset = obj['valid_dataset']
    valid_labels = obj['valid_labels']
    test_dataset = obj['test_dataset']
    test_labels = obj['test_labels']
    del obj

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set   : ', train_dataset.shape, train_labels.shape)
print('Validation set : ', valid_dataset.shape, valid_labels.shape)
print('Test set       : ', test_dataset.shape, test_labels.shape)

hidden_layer1_nodes = 4096
hidden_layer2_nodes = 2048
hidden_layer3_nodes = 1024

training_epochs = 50
starter_learning_rate = 0.05
total_size = train_dataset.shape[0]
batch_size = 100
total_batches = total_size // batch_size
print('Total size : %d, batch size : %d, total batches: %d' % (total_size, batch_size, total_batches))

graph = tf.Graph()
with graph.as_default():

    dataset = tf.placeholder(tf.float32, shape=(None, image_size * image_size))
    labels = tf.placeholder(tf.float32, shape=(None, num_labels))

    layer1_weights = tf.Variable(tf.truncated_normal([image_size * image_size, hidden_layer1_nodes], stddev=0.1))
    layer1_biases = tf.Variable(tf.truncated_normal([hidden_layer1_nodes], stddev=0.1))

    layer2_weights = tf.Variable(tf.truncated_normal([hidden_layer1_nodes, hidden_layer2_nodes], stddev=0.1))
    layer2_biases = tf.Variable(tf.truncated_normal([hidden_layer2_nodes], stddev=0.1))

    layer3_weights = tf.Variable(tf.truncated_normal([hidden_layer2_nodes, hidden_layer3_nodes], stddev=0.1))
    layer3_biases = tf.Variable(tf.truncated_normal([hidden_layer3_nodes], stddev=0.1))

    layer4_weights = tf.Variable(tf.truncated_normal([hidden_layer3_nodes, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.truncated_normal([num_labels], stddev=0.1))

    def model(data, dropout):
        layer1 = tf.matmul(data, layer1_weights) + layer1_biases
        layer1 = tf.nn.relu(layer1)
        layer1 = dropout(layer1)

        layer2 = tf.matmul(layer1, layer2_weights) + layer2_biases
        layer2 = tf.nn.relu(layer2)
        layer2 = dropout(layer2)

        layer3 = tf.matmul(layer2, layer3_weights) + layer3_biases
        layer3 = tf.nn.relu(layer3)
        layer3 = dropout(layer3)

        layer4 = tf.matmul(layer3, layer4_weights) + layer4_biases
        return layer4

    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 10 * total_batches, 0.1, staircase = True)

    l2_loss = 0.001 * (\
        tf.nn.l2_loss(layer1_weights) +\
        tf.nn.l2_loss(layer1_biases) +\
        tf.nn.l2_loss(layer2_weights) +\
        tf.nn.l2_loss(layer2_biases) +\
        tf.nn.l2_loss(layer3_weights) +\
        tf.nn.l2_loss(layer3_biases) +\
        tf.nn.l2_loss(layer4_weights) +\
        tf.nn.l2_loss(layer4_biases))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=model(dataset, lambda dataset: tf.nn.dropout(dataset, 0.5)))) + l2_loss 
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    prediction = tf.nn.softmax(model(dataset, lambda dataset: dataset))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32))

with tf.Session(graph=graph) as session:
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    for epoch in range(training_epochs):
        for batch in range(total_batches):
            choice = np.random.choice(total_size, batch_size)
            feed_dict={dataset: train_dataset[choice], labels: train_labels[choice]}
            _, l, lr = session.run([optimizer, loss, learning_rate], feed_dict)

            if (batch + 1) % 1000 == 0:
                print('Loss at step %d: %f (%f)' % (epoch + 1, l, lr))

        print('Validation accuracy:', accuracy.eval({dataset: valid_dataset, labels: valid_labels}))
        print('Test       accuracy:', accuracy.eval({dataset: test_dataset, labels: test_labels}))
        save_path = saver.save(session, model_file, epoch)
        print("Model saved in file: %s" % save_path)
        print('')