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
decoder_hidden_units = encoder_hidden_units * 2

graph = tf.Graph()
with graph.as_default():

    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    encoder_inputs_length = tf.placeholder(shape=(None), dtype=tf.int32, name='encoder_inputs_length')
    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
    decoder_targets_length = tf.placeholder(shape=(None), dtype=tf.int32, name='decoder_targets_length')
    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
    encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)
    ((encoder_fw_outputs, encoder_bw_outputs),
            (encoder_fw_final_state, encoder_bw_final_state)) = (
                    tf.nn.bidirectional_dynamic_rnn(
                        cell_fw=encoder_cell,
                        cell_bw=encoder_cell,
                        inputs=encoder_inputs_embedded,
                        sequence_length=encoder_inputs_length,
                        dtype=tf.float32, time_major=True)
                    )
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
    encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
    encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
    encoder_final_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_final_state_c, h=encoder_final_state_h)
    decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)
    encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
    decoder_lengths = decoder_targets_length 
    W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
    b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)
    eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
    pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')
    eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
    pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

    def loop_fn_initial():
        initial_elements_finished = (0 >= decoder_lengths)
        initial_input = eos_step_embedded
        initial_cell_state = encoder_final_state
        initial_cell_output = None
        initial_loop_state = None
        return (
                initial_elements_finished,
                initial_input,
                initial_cell_state,
                initial_cell_output,
                initial_loop_state)

    def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

        def get_next_input():
            output_logits = tf.add(tf.matmul(previous_output, W), b)
            prediction = tf.argmax(output_logits, axis=1)
            next_input = tf.nn.embedding_lookup(embeddings, prediction)
            return next_input

        elements_finished = (time >= decoder_lengths)

        finished = tf.reduce_all(elements_finished)
        input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
        state = previous_state
        output = previous_output
        loop_state = None

        return (
                elements_finished,
                input,
                state,
                output,
                loop_state
                )

    def loop_fn(time, previous_output, previous_state, previous_loop_state):
        if previous_state is None:
            assert previous_output is None and previous_state is None
            return loop_fn_initial()
        else:
            return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

    decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
    decoder_outputs = decoder_outputs_ta.stack()
    decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
    decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
    decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
    decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
    decoder_prediction = tf.argmax(decoder_logits, 2)
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
            logits=decoder_logits
            )
    loss = tf.reduce_mean(stepwise_cross_entropy)
    learning_rate = tf.placeholder(tf.float32)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())

    def next_feed(batch):
        encoder_inputs_, encoder_inputs_length_ = major([seq[0] for seq in batch])
        decoder_targets_, decoder_targets_length_ = major([seq[1] + [EOS] for seq in batch])
        return {
                encoder_inputs: encoder_inputs_,
                encoder_inputs_length: encoder_inputs_length_,
                decoder_targets: decoder_targets_,
                decoder_targets_length: decoder_targets_length_
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
