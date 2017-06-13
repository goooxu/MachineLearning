import numpy as np
import tensorflow as tf

def tostring(array):
    return ''.join(chr(i) for i in array)

def major(inputs, max_sequence_length=None):

    sequence_length = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_length)

    inputs_batch_major = np.zeros(shape=(batch_size, max_sequence_length), dtype=np.int32)

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    inputs_batch_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_batch_major, sequence_length

def random_expressions(batch_size):
    opchars = ['+', '-', '*']

    def random_expression(argc):
        args = np.random.randint(10000, size=argc)
        ops = np.random.randint(3, size=argc-1)
        input_exp = '{}'.format(args[0])
        for i in range(argc - 1):
            input_exp += '{}{}'.format(opchars[ops[i]],args[i + 1])
        output_exp = str(eval(input_exp))
        return ([ord(c) for c in input_exp], [ord(c) for c in output_exp])

    while True:
        argcs = np.random.randint(2, 5, batch_size)
        yield [random_expression(i) for i in argcs]

PAD = 0
EOS = 3 

vocab_size = 128 
input_embedding_size = 20 

encoder_hidden_units = 20 
decoder_hidden_units = encoder_hidden_units

graph = tf.Graph()
with graph.as_default():

    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
    decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')
    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)

    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
    encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, dtype=tf.float32, time_major=True)
    del encoder_outputs

    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
    decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell, decoder_inputs_embedded, initial_state=encoder_final_state, dtype=tf.float32, time_major=True, scope="plain_decoder")

    decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)
    decoder_prediction = tf.argmax(decoder_logits, 2)

    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32), logits=decoder_logits)
    loss = tf.reduce_mean(stepwise_cross_entropy)

    learning_rate = tf.placeholder(tf.float32)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    def next_feed(batch):
        encoder_inputs_, _ = major([sequence[0] for sequence in batch])
        decoder_targets_, _ = major([(sequence[1]) + [EOS] for sequence in batch])
        decoder_inputs_, _ = major([[EOS] + (sequence[1]) for sequence in batch])
        return {
            encoder_inputs: encoder_inputs_,
            decoder_inputs: decoder_inputs_,
            decoder_targets: decoder_targets_
        }

    def print_sample(inputs, targets, predicts, count):
        for i, (inp, target, pred) in enumerate(zip(inputs, targets, predicts)):
            print('  sample {}:'.format(i + 1))
            print('    input     > {}'.format(''.join(chr(i) for i in inp)))
            print('    target    > {}'.format(''.join(chr(i) for i in target)))
            print('    predicted > {}'.format(''.join(chr(i) for i in pred)))
            if i >= count - 1:
                break
        print()

    while True:
        command = input('Input your command: ')
        action = command[0]
        args = command[1:].strip().split()

        if action == 't':
            try:
                epoch = int(args[0])
                if len(args) > 1:
                    lr = float(args[1])
                else:
                    lr = 0.001
            except ValueError:
                print('Invalid arguments.')
                continue

            max_steps = 1000 * epoch + 1
            steps_in_epoch = 1000

            batch_size = 100
            batches = random_expressions(batch_size)

            print('Example of the trainning dataset:')
            for seq in next(batches)[:10]:
                print('\t{}={}'.format(tostring(seq[0]), tostring(seq[1])))

            try:
                for batch in range(max_steps):
                    feed = next_feed(next(batches))
                    feed[learning_rate] = lr
                    sess.run(train_op, feed)

                    if batch == 0 or batch % steps_in_epoch == 0:
                        l, predict = sess.run([loss, decoder_prediction], feed)
                        print('batch {}'.format(batch))
                        print('  minibatch loss: {}'.format(l))
                        print_sample(feed[encoder_inputs].T, feed[decoder_targets].T, predict.T, 3)

            except KeyboardInterrupt:
                print('training interrupted')

            print('Training action finished, epoch={}K, learning_rate={}'.format(epoch, lr))

        elif action == 'p':
            try:
                expression = args[0].strip()
            except (ValueError, IndexError):
                print('Invalid arguments.')
                continue

            try:
                ret = str(eval(expression))
            except Exception:
                ret = 'Error!'
            batches = [([ord(i) for i in expression], [ord(i) for i in ret])]
            feed = next_feed(batches)
            predict = sess.run(decoder_prediction, feed)
            print_sample(feed[encoder_inputs].T, feed[decoder_targets].T, predict.T, 1)

        elif action == 'q':
            break