import tensorflow.compat.v1 as tf

class LA_HCN(object):
    def __init__(
            self, sequence_length, num_classes_list, total_classes, vocab_size, lstm_hidden_size,
            attention_unit_size, fc_hidden_size, embedding_size,
            l2_reg_lambda=0.0, concept_dim=50,pretrained_embedding=None):

        self.placeholder_define(sequence_length, num_classes_list, total_classes)
        self.Variable_define(
            num_classes_list=num_classes_list,
            attention_unit_size=attention_unit_size,
            fc_hidden_size=fc_hidden_size,
            concept_num=concept_dim,
            word_emb_size=embedding_size,
            total_classes=total_classes,
            vocab_size=vocab_size,
            pretrained_embedding=pretrained_embedding,
            lstm_hidden_size = lstm_hidden_size)

        self.local_embedding_preprocessing(num_classes_list)
        self.Input_Layer(lstm_hidden_size)

            # First Level
        local_fc_out_mean = None
        self.att_weight_global = []
        self.att_weight_local = []
        self.att_out_global = []
        self.att_out_local = []
        self.label_weight_logit = []
        self.label_weight_scores = []
        self.concept_weight = []
        self.global_fc_out_mean_list = []
        self.local_logits_list = []
        self.local_scores_list = []
        for level_idx, _ in enumerate(num_classes_list):
            Attn_mat_global, output_global, predoc_predict = self.Attention_Layer(
                input_x=self.lstm_out,
                previous_doc=local_fc_out_mean,
                level_i=level_idx)
            global_fc_out = self.FC_Layer(output_global, level_idx, fc_hidden_size)
            if predoc_predict is not None:
                local_fc_out = tf.multiply(global_fc_out, tf.transpose(predoc_predict, perm=[0,2,1]))
            else:
                local_fc_out = global_fc_out

            local_logits, local_scores = self.Local_Predict_Layer(
                input_x = local_fc_out,
                fc_hidden_size = fc_hidden_size,
                level_i = level_idx,
                num_class_list = num_classes_list)
            local_fc_out_mean = tf.reduce_mean(local_fc_out, axis=1)
            global_fc_out_mean = tf.reduce_mean(global_fc_out, axis=1)
            self.global_fc_out_mean_list.append(global_fc_out_mean)
            self.local_logits_list.append(local_logits)
            self.local_scores_list.append(local_scores)

        self.global_fc_out = tf.concat(self.global_fc_out_mean_list, axis=1)

        self.fc_out = self.FC_Layer(
            input_x=self.global_fc_out,
            level_i='global',
            fc_hidden_size = fc_hidden_size)

        self.fc_out = self.Highway_Layer(
            input_x = self.fc_out,
            fc_hidden_size = fc_hidden_size,
            num_layers=1, bias=0)

        with tf.name_scope("dropout"):
            self.fc_out_drop = tf.nn.dropout(self.fc_out, self.dropout)
        with tf.name_scope("global-output"):
            W = tf.Variable(tf.truncated_normal(shape=[fc_hidden_size, total_classes],
                                                stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(tf.constant(value=0.1, shape=[1, total_classes], dtype=tf.float32), name="b")
            self.global_logits = tf.matmul(self.fc_out_drop, W, name="global_logits") + b
            self.global_scores = tf.sigmoid(self.global_logits, name="global_scores")
        with tf.name_scope("output"):
            self.local_scores = tf.concat(self.local_scores_list, axis=1)
            self.scores = tf.add(self.beta * self.global_scores, (1 - self.beta) * self.local_scores, name="combine_scores")

        with tf.name_scope("loss"):
            def cal_loss(labels, logits, name):
                losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
                losses = tf.reduce_mean(tf.reduce_sum(losses, axis=1), name=name + "losses")
                return losses

            # Local Loss
            loss_local_list = []
            for level_idx, num_class in enumerate(num_classes_list):
                loss_local_list.append(
                    cal_loss(labels=self.input_y_local[level_idx],
                             logits=self.local_logits_list[level_idx],
                             name="loss_local_level_{}".format(level_idx)))
            local_losses = tf.add_n(loss_local_list, name="local_losses")
            global_losses = cal_loss(labels=self.input_y, logits=self.global_logits, name="global_")
            l2_losses = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()],
                                 name="l2_losses") * l2_reg_lambda
            self.loss = tf.add_n([global_losses, local_losses, l2_losses], name="loss")


    def placeholder_define(self, sequence_length, num_classes_list, total_classes):
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y_local = []
        for level_idx, label_level in enumerate(num_classes_list):
            self.input_y_local.append(
                tf.placeholder(tf.float32, [None, label_level], name="input_y_{}".format(level_idx)))

        self.input_y = tf.placeholder(tf.float32, [None, total_classes], name="input_y")
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        self.beta = tf.placeholder(tf.float32, name="beta")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

    def Variable_define(self, num_classes_list, attention_unit_size, fc_hidden_size, concept_num, word_emb_size,
                        total_classes, pretrained_embedding, vocab_size, lstm_hidden_size):
        self.concept_features_list = []
        self.W_word_to_concept_list = []
        self.W_pre_doc_to_concept_list = []
        self.W_label_to_concept_list = []
        self.W_pre_doc_to_label_list = []

        # Word Emb Vairable
        if pretrained_embedding is None:
            self.embedding = tf.Variable(
                tf.random_uniform([vocab_size, word_emb_size], minval=-1.0, maxval=1.0, dtype=tf.float32),
                trainable=True, name="embedding")
        else:
            self.embedding = tf.Variable(pretrained_embedding, trainable=True, dtype=tf.float32,name="embedding")

        # Label Variable
        self.label_embedding = tf.Variable(
            tf.random_uniform([total_classes, fc_hidden_size], minval=-1.0, maxval=1.0,
                              dtype=tf.float32), trainable=True, name="label_embedding")

        # Transform Matrix
        for level_idx, num_class in enumerate(num_classes_list):
            self.concept_features_list.append(tf.Variable(
                tf.truncated_normal(shape=[concept_num, attention_unit_size], stddev=0.1, dtype=tf.float32),
                name="level-{}-concept_fea".format(level_idx)))
            self.W_word_to_concept_list.append(tf.Variable(
                tf.truncated_normal(shape=[lstm_hidden_size*2, attention_unit_size], stddev=0.1, dtype=tf.float32),
                name="level-{}-W_word2concept".format(level_idx)))
            self.W_label_to_concept_list.append(tf.Variable(
                tf.truncated_normal(shape=[fc_hidden_size, attention_unit_size], stddev=0.1, dtype=tf.float32),
                name="level-{}-W_label2concept".format(level_idx)))
            if level_idx != 0:
                self.W_pre_doc_to_concept_list.append(tf.Variable(
                    tf.truncated_normal(shape=[fc_hidden_size, attention_unit_size], stddev=0.1, dtype=tf.float32),
                    name="level-{}-W_predoc2concept".format(level_idx)))
                self.W_pre_doc_to_label_list.append(tf.Variable(
                    tf.truncated_normal(shape=[fc_hidden_size, fc_hidden_size], stddev=0.1, dtype=tf.float32),
                    name="level-{}-W_predoc2label".format(level_idx)))

        # Others
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")

    def Highway_Layer(self, input_x, fc_hidden_size, num_layers=1, bias=0.0):

        input_ = input_x
        for idx in range(num_layers):
            with tf.variable_scope('highway_layer_{}'.format(idx)):
                with tf.variable_scope('highway_layer_variable_{}'.format(idx)):
                    W_g = tf.get_variable("W_g", [fc_hidden_size, fc_hidden_size], dtype=input_x.dtype)
                    b_g = tf.get_variable("b_g", [1, fc_hidden_size], dtype=input_x.dtype)
                    W_t = tf.get_variable("W_t", [fc_hidden_size, fc_hidden_size], dtype=input_x.dtype)
                    b_t = tf.get_variable("b_t", [1, fc_hidden_size], dtype=input_x.dtype)
                g = tf.nn.relu(tf.matmul(input_, W_g) + b_g, name = "highway_lin_{}".format(idx))
                t = tf.sigmoid(tf.matmul(input_, W_t) + b_t, name = "highway_gate_{}".format(idx)) + bias
                output = t * g + (1. - t) * input_
                input_ = output

        return output

    def local_embedding_preprocessing(self, num_classes_list):
        self.hier_split = []
        start = 0
        for i in num_classes_list:
            end = start + i
            self.hier_split.append([start, end])
            start = end

        self.label_hier_rel_list = []
        self.label_embedding_list = []
        with tf.name_scope("graph_embedding"):
            for level_idx, _ in enumerate(num_classes_list):
                self.label_embedding_list.append(
                    self.label_embedding[self.hier_split[level_idx][0]:self.hier_split[level_idx][1]])

    def Input_Layer(self, lstm_hidden_size):
        self.embedded_sentence = tf.nn.embedding_lookup(self.embedding, self.input_x)
        self.embedded_sentence_average = tf.reduce_mean(self.embedded_sentence, axis=1)
        # Bi-LSTM Layer
        with tf.name_scope("Bi-lstm"):
            lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)  # forward direction cell
            lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(lstm_hidden_size)  # backward direction cell
            if self.dropout is not None:
                lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=self.dropout)
                lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=self.dropout)
            outputs, state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.embedded_sentence, dtype=tf.float32)
            self.lstm_out = tf.concat(outputs, axis=2)  # [batch_size, sequence_length, lstm_hidden_size * 2]
            self.lstm_out_pool = tf.reduce_mean(self.lstm_out, axis=1)  # [batch_size, lstm_hidden_size * 2]

    def FC_Layer(self, input_x, level_i, fc_hidden_size):
        with tf.name_scope('Fc_layer_{}'.format(level_i)):
            num_units = input_x.get_shape().as_list()[-1]
            W = tf.Variable(
                tf.truncated_normal(shape=[1] * (len(input_x.get_shape().as_list()) - 2) + [num_units, fc_hidden_size],
                                    stddev=0.1, dtype=tf.float32), name="W")
            b = tf.Variable(
                tf.constant(value=0.1, shape=[1] * (len(input_x.get_shape().as_list()) - 2) + [1, fc_hidden_size],
                            dtype=tf.float32), name="b")
            fc = tf.matmul(input_x, W) + b
            fc_out = tf.nn.relu(fc)
        return fc_out

    def Attention_Layer(self, input_x, previous_doc, level_i):
        with tf.name_scope('Attention_layer_{}'.format(level_i)):
            label_in_concept = tf.tanh(
                tf.matmul(self.label_embedding_list[level_i],
                          self.W_label_to_concept_list[level_i], name='label_in_concept'))
            R_mat = tf.nn.softmax(
                tf.matmul(label_in_concept, self.concept_features_list[level_i], transpose_b=True), axis=-1, name='R_mat') # (L, C)

            if level_i == 0:
                word_in_concept = tf.tanh(tf.matmul(
                    input_x,tf.expand_dims(self.W_word_to_concept_list[level_i], axis=0)), name='word_in_concept')
                S_mat = tf.matmul(
                    word_in_concept,
                    tf.expand_dims(tf.transpose(self.concept_features_list[level_i], perm=[1, 0]), axis=0), name='S_mat')  # (1, W, C)
                Attn_mat_global = tf.nn.softmax(
                    tf.matmul(tf.expand_dims(R_mat, axis=0), tf.transpose(S_mat, perm=[0, 2, 1])), axis=-1,
                    name='Attn_mat_global')  # (1, L, W)
                predoc_predict = None
            else:
                # concatenate word_emb with pre_doc_emb
                word_in_concept = tf.tanh(
                    tf.matmul(input_x, tf.expand_dims(self.W_word_to_concept_list[level_i], axis=0))
                    + tf.expand_dims(tf.matmul(previous_doc, self.W_pre_doc_to_concept_list[level_i-1]),axis=1),
                    name='word_in_concept')
                S_mat = tf.matmul(
                    word_in_concept,
                    tf.expand_dims(tf.transpose(self.concept_features_list[level_i], perm=[1, 0]), axis=0), name='S_mat')  # (1, W, C)
                Attn_mat_global = tf.nn.softmax(
                    tf.matmul(tf.expand_dims(R_mat, axis=0), tf.transpose(S_mat, perm=[0, 2, 1])), axis=-1,
                    name='Attn_mat_global')  # (1, L, W)

                predoc_in_label = tf.expand_dims(tf.tanh(
                    tf.matmul(previous_doc, self.W_pre_doc_to_label_list[level_i-1])), axis =1)
                predoc_predict = tf.nn.leaky_relu(tf.matmul(predoc_in_label,
                                           tf.expand_dims(
                                               tf.transpose(self.label_embedding_list[level_i], perm=[1,0]), axis=0)))
                predoc_predict = tf.nn.sigmoid(predoc_predict, name='predoc_predict')
            output_global = tf.matmul(Attn_mat_global, input_x, name='output_global')
        return Attn_mat_global, output_global, predoc_predict

    def Local_Predict_Layer(self, input_x, fc_hidden_size, level_i, num_class_list):
        with tf.name_scope('Local_Predict_Layer_{}'.format(level_i)):
            num_units = input_x.get_shape().as_list()[-1]
            W = tf.Variable(tf.truncated_normal(shape=[1, num_units, fc_hidden_size], stddev=0.1, dtype=tf.float32), name="W")
            fc = tf.matmul(input_x, W)
            b = tf.Variable(tf.constant(value=0.1, shape=[1, num_class_list[level_i]], dtype=tf.float32), name="b")
            logits = tf.reduce_sum(tf.multiply(fc, tf.expand_dims(self.label_embedding_list[level_i], axis=0)), axis=-1) + b
            scores = tf.sigmoid(logits, name="scores")
        return logits, scores