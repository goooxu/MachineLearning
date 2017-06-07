import numpy as np
import tensorflow as tf
from six.moves import cPickle

image_size = 28
num_labels = 10
num_channels = 1
pickle_file = 'notMNIST.pickle'

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
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set   : ', train_dataset.shape, train_labels.shape)
print('Validation set : ', valid_dataset.shape, valid_labels.shape)
print('Test set       : ', test_dataset.shape, test_labels.shape)

training_epochs = 50
starter_learning_rate = 0.05
total_size = train_dataset.shape[0]
batch_size = 100
total_batches = total_size // batch_size

layer1_patch_size = 9
layer1_depth = 16
layer2_patch_size = 5
layer2_depth = 50 
num_hidden = 1024
print('Total size : %d, batch size : %d, total batches: %d' % (total_size, batch_size, total_batches))

graph = tf.Graph()
with graph.as_default():

    dataset = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
    labels = tf.placeholder(tf.float32, shape=(None, num_labels))

    layer1_weights = tf.Variable(tf.truncated_normal([layer1_patch_size, layer1_patch_size, num_channels, layer1_depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([layer1_depth]))

    layer2_weights = tf.Variable(tf.truncated_normal([layer2_patch_size, layer2_patch_size, layer1_depth, layer2_depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[layer2_depth]))

    layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * layer2_depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

    def model(data, dropout):
        layer1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], 'SAME')
        layer1 = tf.nn.relu(layer1 + layer1_biases)
        layer1 = tf.nn.max_pool(layer1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        layer1 = dropout(layer1)

        layer2 = tf.nn.conv2d(layer1, layer2_weights, [1, 1, 1, 1], 'SAME')
        layer2 = tf.nn.relu(layer2 + layer2_biases)
        layer2 = tf.nn.max_pool(layer2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        layer2 = dropout(layer2)

        shape = tf.shape(layer2)
        layer3 = tf.reshape(layer2, [shape[0], shape[1] * shape[2] * shape[3]])
        layer3 = tf.nn.relu(tf.matmul(layer3, layer3_weights) + layer3_biases)
        layer3 = dropout(layer3)

        layer4 = tf.matmul(layer3, layer4_weights) + layer4_biases
        return layer4

    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 15 * total_batches, 0.1, staircase = True)

    l2_loss = 0.001 * (\
        tf.nn.l2_loss(layer1_weights) +\
        tf.nn.l2_loss(layer1_biases) +\
        tf.nn.l2_loss(layer2_weights) +\
        tf.nn.l2_loss(layer2_biases) +\
        tf.nn.l2_loss(layer3_weights) +\
        tf.nn.l2_loss(layer3_biases) +\
        tf.nn.l2_loss(layer4_weights) +\
        tf.nn.l2_loss(layer4_biases))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=model(dataset, lambda dataset: tf.nn.dropout(dataset, 0.75)))) + l2_loss
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    prediction = tf.nn.softmax(model(dataset, lambda dataset: dataset))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)), tf.float32))

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()

    for epoch in range(training_epochs):
        for batch in range(total_batches):
            choice = np.random.choice(total_size, batch_size)
            feed_dict = {dataset : train_dataset[choice], labels : train_labels[choice]}
            _, l, lr = session.run([optimizer, loss, learning_rate], feed_dict)
            
            if ((batch + 1) % 1000 == 0):
                print('Loss at step %d: %f (%f)' % (epoch + 1, l, lr))

        print('Validation  accuracy: %f' % accuracy.eval({dataset : valid_dataset, labels : valid_labels}))
        print('Test        accuracy: %f' % accuracy.eval({dataset : test_dataset, labels : test_labels}))
