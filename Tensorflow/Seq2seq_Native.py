import numpy as np
import tensorflow as tf
import math

class Seq2SeqModel():
    PAD = 0
    EOS = 3

    def __init__(self, encoder_cell, decoder_cell, vocab_size, embedding_size, bidirectional=True, attention=False):
        self.bidirectional = bidirectional
        self.attention = attention
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.encoder_cell = encoder_cell
        self.decoder_cell = decoder_cell
        self._make_graph()

    @property
    def decoder_hidden_units(self):
        return self.decoder_cell.output_size

    def _make_graph(self):
        self._init_placeholders()

        self._init_decoder_train_connectors()
        self._init_embeddings()

        if self.bidirectional:
            self._init_bidirectional_encoder()
        else:
            self._init_simple_encoder()

        self._init_decoder()
        self._init_optimizer()

    def _init_placeholders(self):

        self.encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(shape=(None), dtype=tf.int32, name='encoder_inputs_length')
        self.decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(shape=(None), dtype=tf.int32, name='decoder_targets_length')

    def _init_decoder_train_connectors(self):

        with tf.name_scope('DecoderTrainFeeds'):
            sequence_size, batch_size = tf.unstack(tf.shape(self.decoder_targets))

            EOS_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.EOS
            PAD_SLICE = tf.ones([1, batch_size], dtype=tf.int32) * self.PAD

            self.decoder_train_inputs = tf.concat([EOS_SLICE, self.decoder_targets], axis=0)
            self.decoder_train_length = self.decoder_targets_length + 1

            decoder_train_targets = tf.concat([self.decoder_targets, PAD_SLICE], axis=0)
            decoder_train_targets_seq_len, _ = tf.unstack(tf.shape(decoder_train_targets))
            decoder_train_targets_eos_mask = tf.one_hot(
                self.decoder_train_length - 1,
                decoder_train_targets_seq_len,
                on_value=self.EOS, off_value=self.PAD, dtype=tf.int32)
            decoder_train_targets_eos_mask = tf.transpose(decoder_train_targets_eos_mask, [1, 0])
            decoder_train_targets = tf.add(decoder_train_targets, decoder_train_targets_eos_mask)

            self.decoder_train_targets = decoder_train_targets
            self.loss_weights = tf.ones([
                batch_size,
                tf.reduce_max(self.decoder_train_length)
            ], dtype=tf.float32, name='loss_weights')

    def _init_embeddings(self):
        
        with tf.variable_scope('Embedding') as scope:

            sqrt3 = math.sqrt(3)
            initializer = tf.random_uniform_initializer(-sqrt3, sqrt3)

            self.embedding_matrix = tf.get_variable(
                name='embedding_matrix',
                shape=(self.vocab_size, self.embedding_size),
                initializer=initializer,
                dtype=tf.float32)

            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.encoder_inputs)
            self.decoder_train_inputs_embedded = tf.nn.embedding_lookup(self.embedding_matrix, self.decoder_train_inputs)

    def _init_simple_encoder(self):

        with tf.variable_scope('Encoder') as scope:
            (self.encoder_outputs, self.encoder_state) = (
                tf.nn.dynamic_rnn(
                    cell=self.encoder_cell,
                    inputs=self.encoder_inputs_embedded,
                    sequence_length=self.encoder_inputs_length,
                    time_major=True,
                    dtype=tf.float32)
            )

    def _init_bidirectional_encoder(self):

        with tf.variable_scope('BidirectionalEncoder') as scope:

            ((encoder_fw_outputs, encoder_bw_outputs),
            (encoder_fw_state, encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.encoder_cell,
                    cell_bw=self.encoder_cell,
                    inputs=self.encoder_inputs_embedded,
                    sequence_length=self.encoder_inputs_length,
                    time_major=True,
                    dtype=tf.float32
                )
            )

            self.encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
            
            if isinstance(encoder_fw_state, tf.contrib.rnn.LSTMStateTuple):

                encoder_state_c = tf.concat((encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
                encoder_state_h = tf.concat((encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
                self.encoder_state = tf.contrib.rnn.LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)

            elif isinstance(encoder_fw_state, tf.Tensor):
                self.encoder_state = tf.concat((encoder_fw_state, encoder_bw_state), 1, name='bidirectional_concat')

    def _init_decoder(self):
        
        with tf.variable_scope('Decoder') as scope:

            def output_fn(outputs):
                return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)
            
            if not self.attention:
                decoder_fn_train = seq2seq.simple_decoder_fn_train(encoder_state=self.encoder_state)
                decoder_fn_inference = seq2seq.simple_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    embeddings=self.embeddingmatrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size)
            else:
                attention_states = tf.transpose(self.encoder_outputs, [1, 0, 2])

                (attention_keys, attention_values, attention_score_fn, attention_construct_fn) = tf.contrib.seq2seq.prepare_attention(
                    attention_states=attention_states,
                    attention_option='bahdanau',
                    num_units=self.decoder_hidden_units
                )

                decoder_fn_train = tf.contrib.seq2seq.attention_decoder_fn_train(
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    name='attention_decoder'
                )

                decoder_fn_inference = tf.contrib.seq2seq.attention_decoder_fn_inference(
                    output_fn=output_fn,
                    encoder_state=self.encoder_state,
                    attention_keys=attention_keys,
                    attention_values=attention_values,
                    attention_score_fn=attention_score_fn,
                    attention_construct_fn=attention_construct_fn,
                    embeddings=self.embedding_matrix,
                    start_of_sequence_id=self.EOS,
                    end_of_sequence_id=self.EOS,
                    maximum_length=tf.reduce_max(self.encoder_inputs_length) + 3,
                    num_decoder_symbols=self.vocab_size
                )

                (self.decoder_outputs_train,
                self.decoder_state_train,
                self.decoder_context_state_train) = (
                    tf.contrib.seq2seq.dynamic_rnn_decoder(
                        cell=self.decoder_cell,
                        decoder_fn=decoder_fn_train,
                        inputs=self.decoder_train_inputs_embedded,
                        sequence_length=self.decoder_train_length,
                        time_major=True,
                        scope=scope
                    )
                )

                self.decoder_logits_train = output_fn(self.decoder_outputs_train)
                self.decoder_prediction_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_prediction_train')

                scope.reuse_variables()

                (self.decoder_logits_inference,
                self.decoder_state_inference,
                self.decoder_context_state_inference) = (
                    tf.contrib.seq2seq.dynamic_rnn_decoder(
                        cell=self.decoder_cell,
                        decoder_fn=decoder_fn_inference,
                        time_major=True,
                        scope=scope
                    )
                )

                self.decoder_prediction_inference = tf.argmax(self.decoder_logits_inference, axis=-1, name='decoder_prediction_inference')

    def _init_optimizer(self):

        logits = tf.transpose(self.decoder_logits_train, [1, 0, 2])
        targets = tf.transpose(self.decoder_train_targets, [1, 0])
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=logits, targets=targets, weights=self.loss_weights)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def _major(self, inputs, max_sequence_length=None):

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

    def make_train_inputs(self, input_seq, target_seq):
        inputs_, inputs_length_ = self._major(input_seq)
        targets_, targets_length_ = self._major(target_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_,
            self.decoder_targets: targets_,
            self.decoder_targets_length: targets_length_
        }

    def make_inference_inputs(self, input_seq):
        inputs_, inputs_length_ = self._major(input_seq)
        return {
            self.encoder_inputs: inputs_,
            self.encoder_inputs_length: inputs_length_
        }

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

def print_sample(inputs, targets, predicteds, count):
    for i, (inp, target, pred) in enumerate(zip(inputs, targets, predicteds)):
        print('  sample {}:'.format(i + 1))
        print('    input     > {}'.format(''.join(chr(i) for i in inp)))
        print('    target    > {}'.format(''.join(chr(i) for i in target)))
        print('    predicted > {}'.format(''.join(chr(i) for i in pred)))
        if i >= count - 1:
            break
    print()

batch_size=128
batches_in_epoch = 1000

with tf.Session() as session:
    
    model = Seq2SeqModel(
        encoder_cell=tf.contrib.rnn.LSTMCell(10),
        decoder_cell=tf.contrib.rnn.LSTMCell(20),
        vocab_size=128,
        embedding_size=10,
        attention=True,
        bidirectional=True)

    session.run(tf.global_variables_initializer())

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

            dataset_iter = random_expressions(batch_size)

            try:
                for batch in range(epoch * 1000 + 1):
                    batch_data = next(dataset_iter)
                    fd = model.make_train_inputs([seq[0] for seq in batch_data], [seq[1] for seq in batch_data])
                    _, l = session.run([model.train_op, model.loss], fd)

                    if batch % batches_in_epoch == 0:
                        print('batch {}'.format(batch))
                        print('  minimatch loss: {}'.format(session.run(model.loss, fd)))
                        print_sample(fd[model.encoder_inputs].T, fd[model.decoder_targets].T, session.run(model.decoder_prediction_train, fd).T, 3)

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

            fd = model.make_inference_inputs([[ord(i) for i in expression]])
            print_sample(fd[model.encoder_inputs].T, [[ord(i) for i in ret]], session.run(model.decoder_prediction_inference, fd).T, 1)

        elif action == 'q':
            break
