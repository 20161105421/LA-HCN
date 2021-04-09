import os, time, pickle, argparse
import numpy as np
import tensorflow.compat.v1 as tf
from src.model import LA_HCN
from utils import checkmate as cm
from utils import data_helpers as dh

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score


def parse_args():
    # Parameters
    # ==================================================
    parser = argparse.ArgumentParser(description="Run LA_HCN.")

    # hyper-para for datasets
    parser.add_argument('--dataname', type=str, default='enron_0', help="training data.")
    parser.add_argument('--training_data_file', type=str, default='data/data_16_nov/enron/0/enron_train_0.json', help="path to training data.")
    parser.add_argument('--validation_data_file', type=str, default='data/data_16_nov/enron/0/enron_val_0.json', help="path to validation data.")
    parser.add_argument('--num_classes_list', type=str, default="3,40,13", help="Number of labels list (depends on the task)")
    parser.add_argument('--glove_file', type=str, default="data/data_16_nov/glove6b100dtxt/glove.6B.100d.txt", help="glove embeding file")
    parser.add_argument('--train_or_restore', type=str, default='Train', help="Train or Restore. (default: Train)")

    # hyper-para for training
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning Rate.")
    parser.add_argument('--batch_size', type=int, default=30, help="Batch Size (default: 256)")
    parser.add_argument('--num_epochs', type=int, default=100, help="Number of training epochs (default: 100)")
    parser.add_argument('--pad_seq_len', type=int, default=250, help="Recommended padding Sequence length of data (depends on the data)")
    parser.add_argument('--embedding_dim', type=int, default=100,help="Dimensionality of character embedding (default: 128)")
    parser.add_argument('--lstm_hidden_size', type=int, default=256,
                        help="Hidden size for bi-lstm layer(default: 256)")
    parser.add_argument('--attention_unit_size', type=int, default=200,
                        help="Attention unit size(default: 200)")
    parser.add_argument('--fc_hidden_size', type=int, default=512,
                        help="Hidden size for fully connected layer (default: 512)")
    parser.add_argument('--dropout', type=float, default=0.5, help= "Dropout keep probability (default: 0.5)")
    parser.add_argument('--l2_reg_lambda', type=float, default= 0.0, help="L2 regularization lambda (default: 0.0)")
    parser.add_argument('--beta', type=float, default=0.5, help="Weight of global scores in scores cal")
    parser.add_argument('--norm_ratio', type=float, default=2, help="The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
    parser.add_argument('--decay_steps', type=int, default=5000,
                        help="The ratio of the sum of gradients norms of trainable variable (default: 1.25)")
    parser.add_argument('--decay_rate', type=float, default=0.95, help="Rate of decay for learning rate. (default: 0.95)")
    parser.add_argument('--checkpoint_every', type=int, default=100, help="Save model after this many steps (default: 100)")
    parser.add_argument('--num_checkpoints', type=int, default=5, help="Number of checkpoints to store (default: 5)")

    # hyper-para for prediction
    parser.add_argument('--evaluate_every', type=int, default=100, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument('--top_num', type=int, default=5, help="Number of top K prediction classes (default: 5)")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for prediction classes (default: 0.5)")


    parser.set_defaults(directed=False)

    return parser.parse_args()

def train(args):
    # Load sentences, labels, and training parameters
    print("✔︎ Loading data...")
    VOCAB_SIZE, pretrained_glove_emb, word2id_dict = dh.load_glove_word_embedding(args.embedding_dim, args.glove_file)

    if pretrained_glove_emb is None:
        increase_word2id = True
        if args.train_or_restore == 'R':
            word2id_dict = \
                pickle.load(open(os.path.join(os.path.dirname(args.training_data_file), 'word2id_dict.pkl'), 'rb'))
    else:
        increase_word2id = False
    print("✔︎ Training data processing...")
    train_data, word2id_dict = dh.load_data_and_labels(
        data_file = args.training_data_file,
        num_classes_list = args.num_classes_list,
        embedding_size=args.embedding_dim,
        word2id_dict=word2id_dict,
        increase_word2id=increase_word2id)

    print("✔︎ Validation data processing...")
    val_data, word2id_dict = dh.load_data_and_labels(
        data_file = args.validation_data_file,
        num_classes_list=args.num_classes_list,
        embedding_size=args.embedding_dim,
        word2id_dict=word2id_dict,
        increase_word2id=increase_word2id)
    if pretrained_glove_emb is None:
        pickle.dump(word2id_dict,
                    open(os.path.join(os.path.dirname(args.training_data_file), 'word2id_dict.pkl'), 'wb'))

    VOCAB_SIZE = len(word2id_dict)

    # Build a graph and LA-HCN object
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            la_hcn = LA_HCN(
                sequence_length=args.pad_seq_len,
                num_classes_list=list(map(int, args.num_classes_list.split(','))),
                total_classes=sum(list(map(int, args.num_classes_list.split(',')))),
                vocab_size=VOCAB_SIZE,
                lstm_hidden_size=args.lstm_hidden_size,
                attention_unit_size = args.attention_unit_size,
                fc_hidden_size=args.fc_hidden_size,
                embedding_size=args.embedding_dim,
                l2_reg_lambda=args.l2_reg_lambda,
                pretrained_embedding=pretrained_glove_emb)

            # Define training procedure
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                learning_rate = tf.train.exponential_decay(learning_rate=args.learning_rate,
                                                           global_step=la_hcn.global_step, decay_steps=args.decay_steps,
                                                           decay_rate=args.decay_rate, staircase=True)
                optimizer = tf.train.AdamOptimizer(learning_rate)
                grads, vars = zip(*optimizer.compute_gradients(la_hcn.loss))
                grads, _ = tf.clip_by_global_norm(grads, clip_norm=args.norm_ratio)
                train_op = optimizer.apply_gradients(zip(grads, vars), global_step=la_hcn.global_step, name="train_op")

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, vars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{0}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{0}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            if args.train_or_restore == 'R':
                MODEL = input("☛ Please input the checkpoints model you want to restore, "
                              "it should be like(1490175368_{dataname}_{beta}): ")  # The model you want to restore
                print("✔︎ Now loading to next step...")
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", MODEL))
                print("✔︎ Writing to {0}\n".format(out_dir))
            else:
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", '_'.join([timestamp, args.dataname, str(args.beta)])))
                print("✔︎ Writing to {0}\n".format(out_dir))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            best_checkpoint_dir = os.path.abspath(os.path.join(out_dir, "bestcheckpoints"))

            # Summaries for loss
            loss_summary = tf.summary.scalar("loss", la_hcn.loss)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Validation summaries
            validation_summary_op = tf.summary.merge([loss_summary])
            validation_summary_dir = os.path.join(out_dir, "summaries", "validation")
            validation_summary_writer = tf.summary.FileWriter(validation_summary_dir, sess.graph)

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.num_checkpoints)
            best_saver = cm.BestCheckpointSaver(save_dir=best_checkpoint_dir, num_to_keep=3, maximize=True)

            if args.train_or_restore == 'R':
                print("✔︎ Loading model...")
                checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
                print(checkpoint_file)
                saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            else:
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                # Save the embedding visualization
                saver.save(sess, os.path.join(out_dir, "embedding", "embedding.ckpt"))

            current_step = sess.run(la_hcn.global_step)

            def train_step(x_batch, y_batch, y_batch_tuple):
                feed_dict_local_y = dict()
                for idx, y_batch_local in enumerate(y_batch_tuple):
                    feed_dict_local_y[la_hcn.input_y_local[idx]] = y_batch_local
                feed_dict = {
                    la_hcn.input_x: x_batch,
                    la_hcn.input_y: y_batch,
                    la_hcn.dropout: args.dropout,
                    la_hcn.beta: args.beta,
                    la_hcn.is_training: True
                }
                feed_dict.update(feed_dict_local_y)
                _, step, summaries, loss = sess.run(
                    [train_op, la_hcn.global_step, train_summary_op, la_hcn.loss], feed_dict)
                print("step {0}: loss {1:g}".format(step, loss))
                train_summary_writer.add_summary(summaries, step)

            def validation_step(x_val_data, writer=None):
                """Evaluates model on a validation set"""
                batches_validation = dh.batch_iter(
                    list(zip(x_val_data.content_tokenindex, x_val_data.labels, x_val_data.labels_tuple)),
                    args.batch_size,
                    1,
                    args.pad_seq_len,
                    list(map(int, args.num_classes_list.split(','))),
                    sum(list(map(int, args.num_classes_list.split(',')))))

                # Predict classes by threshold or topk ('ts': threshold; 'tk': topk)
                eval_counter, eval_loss = 0, 0.0

                eval_pre_tk = [0.0] * args.top_num
                eval_rec_tk = [0.0] * args.top_num
                eval_F_tk = [0.0] * args.top_num

                true_onehot_labels = []
                predicted_onehot_scores = []
                predicted_onehot_labels_ts = []
                predicted_onehot_labels_tk = [[] for _ in range(args.top_num)]

                for x_batch_val, y_batch_val, y_batch_val_tuple in batches_validation:
                    feed_dict_local_y = dict()
                    for idx, y_batch_local in enumerate(y_batch_val_tuple):
                        feed_dict_local_y[la_hcn.input_y_local[idx]] = y_batch_local
                    feed_dict = {
                        la_hcn.input_x: x_batch_val,
                        la_hcn.input_y: y_batch_val,
                        la_hcn.dropout: 1.0,
                        la_hcn.beta: args.beta,
                        la_hcn.is_training: False
                    }
                    feed_dict.update(feed_dict_local_y)
                    step, summaries, scores, cur_loss = sess.run(
                        [la_hcn.global_step, validation_summary_op, la_hcn.scores, la_hcn.loss], feed_dict)

                    # Predict by threshold
                    batch_predicted_onehot_labels_ts =\
                        dh.get_onehot_label_threshold(scores=scores, threshold=args.threshold)
                    predicted_onehot_labels_ts.append(batch_predicted_onehot_labels_ts)

                    # Predict by topK
                    for top_num in range(args.top_num):
                        batch_predicted_onehot_labels_tk = dh.get_onehot_label_topk(scores=scores, top_num=top_num+1)
                        predicted_onehot_labels_tk[top_num].append(batch_predicted_onehot_labels_tk)

                    true_onehot_labels.append(y_batch_val)
                    predicted_onehot_scores.append(scores)
                    eval_loss = eval_loss + cur_loss
                    eval_counter = eval_counter + 1

                    if writer:
                        writer.add_summary(summaries, step)

                eval_loss = float(eval_loss / eval_counter)

                # Calculate Precision & Recall & F1
                eval_pre_ts = precision_score(y_true=np.vstack(true_onehot_labels),
                                              y_pred=np.vstack(predicted_onehot_labels_ts), average='micro')
                eval_rec_ts = recall_score(y_true=np.vstack(true_onehot_labels),
                                           y_pred=np.vstack(predicted_onehot_labels_ts), average='micro')
                eval_F_ts = f1_score(y_true=np.vstack(true_onehot_labels),
                                     y_pred=np.vstack(predicted_onehot_labels_ts), average='micro')
                eval_auc = roc_auc_score(y_true=np.vstack(true_onehot_labels),
                                         y_score=np.vstack(predicted_onehot_scores), average='micro')
                eval_prc = average_precision_score(y_true=np.vstack(true_onehot_labels),
                                                   y_score=np.vstack(predicted_onehot_scores), average='micro')

                for top_num in range(args.top_num):
                    eval_pre_tk[top_num] = precision_score(y_true=np.vstack(true_onehot_labels),
                                                           y_pred=np.vstack(predicted_onehot_labels_tk[top_num]),
                                                           average='micro')
                    eval_rec_tk[top_num] = recall_score(y_true=np.vstack(true_onehot_labels),
                                                        y_pred=np.vstack(predicted_onehot_labels_tk[top_num]),
                                                        average='micro')
                    eval_F_tk[top_num] = f1_score(y_true=np.vstack(true_onehot_labels),
                                                  y_pred=np.vstack(predicted_onehot_labels_tk[top_num]),
                                                  average='micro')

                return eval_loss, eval_auc, eval_prc, eval_rec_ts, eval_pre_ts, eval_F_ts, \
                       eval_rec_tk, eval_pre_tk, eval_F_tk

            # Generate batches
            batches_train = dh.batch_iter(
                list(zip(train_data.content_tokenindex, train_data.labels, train_data.labels_tuple)),
                args.batch_size,
                args.num_epochs,
                args.pad_seq_len,
                list(map(int, args.num_classes_list.split(','))),
                sum(list(map(int, args.num_classes_list.split(',')))))

            num_batches_per_epoch = int((len(train_data.content_tokenindex) - 1) / args.batch_size) + 1

            # Training loop. For each batch...
            for x_batch_train, y_batch_train, y_batch_train_tuple in batches_train:
                train_step(x_batch_train, y_batch_train, y_batch_train_tuple)
                current_step = tf.train.global_step(sess, la_hcn.global_step)

                if current_step % args.evaluate_every == 0:
                    print("\nEvaluation:")
                    eval_loss, eval_auc, eval_prc, \
                    eval_rec_ts, eval_pre_ts, eval_F_ts, eval_rec_tk, eval_pre_tk, eval_F_tk = \
                        validation_step(val_data, writer=validation_summary_writer)

                    print("All Validation set: Loss {0:g} | AUC {1:g} | AUPRC {2:g}".format(eval_loss, eval_auc, eval_prc))
                    print("☛ Predict by threshold: Precision {0:g}, Recall {1:g}, F {2:g}".format(eval_pre_ts, eval_rec_ts, eval_F_ts))
                    print("☛ Predict by topK:")
                    for top_num in range(args.top_num):
                        print("Top{0}: Precision {1:g}, Recall {2:g}, F {3:g}"
                                    .format(top_num+1, eval_pre_tk[top_num], eval_rec_tk[top_num], eval_F_tk[top_num]))
                    best_saver.handle(eval_prc, sess, current_step)
                if current_step % args.checkpoint_every == 0:
                    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("✔︎ Saved model checkpoint to {0}\n".format(path))
                if current_step % num_batches_per_epoch == 0:
                    current_epoch = current_step // num_batches_per_epoch
                    print("✔︎ Epoch {0} has finished!".format(current_epoch))

    print("✔︎ Done.")


if __name__ == '__main__':
    args = parse_args()
    train(args)