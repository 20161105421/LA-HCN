import os, argparse, pickle
import tensorflow.compat.v1 as tf
from scipy.sparse import lil_matrix

from utils import checkmate as cm
from src.test_utils import *
from utils import data_helpers as dh
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
MODEL = input("☛ Please input the model file you want to test, it should be like(1490175368_{dataname}_{beta}): ")  # The model you want to restore
print("✔︎ The format of your input is legal, now loading to next step...")

def parse_args():
    parser = argparse.ArgumentParser(description="Run LA_HCN.")

    # hyper-para for datasets
    parser.add_argument('--dataname', type=str, default='reuters_0', help="dataname.")
    parser.add_argument('--test_data_file', type=str, default='data/data_16_nov/reuters/0/reuters_test_0.json', help="test data.")
    parser.add_argument('--num_classes_list', type=str, default="4,55,42", help="Number of labels list (depends on the task)")
    parser.add_argument('--glove_file', type=str, default="data/glove6b100dtxt/glove.6B.100d.txt_", help="glove embeding file")
    parser.add_argument('--train_or_restore', type=str, default='Restore', help="Train or Restore.")

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


def evaluate(true_onehot_labels, true_onehot_labels_level_dict,
             predicted_onehot_labels_ts, predicted_onehot_labels_ts_level,
             predicted_onehot_labels_tk,
             predict_scores, predict_scores_level,
             num_classes_list, emb_type, logger):
    pre_lsit = []; rec_list = []; F_list = []; prc_list = []; auc_list = []
    pre_list_level = []; rec_list_level = []; F_list_level = []; prc_list_level = []; auc_list_level = []
    pre_tk = []; rec_tk = []; F_tk = []
    for type in ['macro', 'micro']:
        logger.print("✔︎ #############################   For All Label ({}). ###############################".format(emb_type))
        pre =  precision_score(y_true=true_onehot_labels, y_pred=predicted_onehot_labels_ts, average=type)
        rec = recall_score(y_true=true_onehot_labels, y_pred=predicted_onehot_labels_ts, average=type)
        F = f1_score(y_true=true_onehot_labels, y_pred=predicted_onehot_labels_ts, average=type)
        try:
            prc = average_precision_score(y_true=true_onehot_labels.toarray(), y_score=np.vstack(predict_scores), average=type)
        except:
            prc = 0
        try:
            auc = roc_auc_score(y_true=true_onehot_labels.toarray(), y_score=np.vstack(predict_scores), average=type)
        except:
            auc = 0
        logger.print("☛ All Test Dataset For ALL Label: AUC_{0} {1:g} | AUPRC_{2} {3:g}".format(type, auc, type, prc))
        logger.print("☛ All Test Dataset For All Label: Pre_{0} {1:g} | Rec_{2} {3:g} | F1_{4} {5:g}".format(
            type, pre, type, rec, type, F))
        test_pre_list = []; test_rec_list = []; test_F_list = []; test_prc_list = []; test_auc_list = []
        for i, num_class in enumerate(num_classes_list):
            test_pre_list.append(precision_score(y_true=true_onehot_labels_level_dict[i],
                                                 y_pred=predicted_onehot_labels_ts_level[i],
                                                 average=type))
            test_rec_list.append(recall_score(y_true=true_onehot_labels_level_dict[i],
                                              y_pred=predicted_onehot_labels_ts_level[i],
                                              average=type))
            test_F_list.append(f1_score(y_true=true_onehot_labels_level_dict[i],
                                        y_pred=predicted_onehot_labels_ts_level[i], average=type))
            try:
                test_prc_list.append(average_precision_score(y_true=true_onehot_labels_level_dict[i].toarray(),
                                                             y_score=np.vstack(predict_scores_level[i]), average=type))
            except:
                test_prc_list.append(0)
            try:
                test_auc_list.append(roc_auc_score(y_true=true_onehot_labels_level_dict[i].toarray(),
                                                   y_score=np.vstack(predict_scores_level[i]), average=type))
            except:
                test_auc_list.append(0)
            logger.print(
                "☛ Predict by threshold in Level-{0}: Pre_{1} {2:g}, Rec_{3} {4:g}, F1_{5} {6:g}, AUPRC_{7} {8:g}, AUC_{9} {10:g}".format(
                    i + 1, type, test_pre_list[i], type, test_rec_list[i], type, test_F_list[i], type, test_prc_list[i],
                    type, test_auc_list[i]))

        test_pre_tk = []; test_rec_tk = []; test_F_tk = []
        for level_i in range(args.top_num):
            test_pre_tk.append(precision_score(y_true=true_onehot_labels,
                                               y_pred=predicted_onehot_labels_tk[level_i], average=type))
            test_rec_tk.append(recall_score(y_true=true_onehot_labels,
                                            y_pred=predicted_onehot_labels_tk[level_i], average=type))
            test_F_tk.append(f1_score(y_true=true_onehot_labels,
                                      y_pred=predicted_onehot_labels_tk[level_i], average=type))
            logger.print("☛ Predict by topk-{0:g}: Pre_{1} {2:g}, Rec_{3} {4:g}, F1_{5} {6:g}".format(level_i + 1, type,
                                                                                               test_pre_tk[level_i],
                                                                                               type,
                                                                                               test_rec_tk[level_i],
                                                                                               type,
                                                                                               test_F_tk[level_i]))

            pre_tk.append(test_pre_tk); rec_tk.append(test_rec_tk); F_tk.append(test_F_tk)
        logger.print('\n')
    return pre_lsit, rec_list, F_list, prc_list, auc_list, \
           pre_list_level, rec_list_level, F_list_level, prc_list_level, auc_list_level, \
           pre_tk, rec_tk, F_tk

def test_lahcn(args):

    # Load data
    print("✔︎ Loading data...")
    print("Recommended padding Sequence length is: {0}".format(args.pad_seq_len))

    print("✔︎ Test data processing...")
    VOCAB_SIZE, pretrained_glove_emb, word2id_dict = dh.load_glove_word_embedding(args.embedding_dim, args.glove_file)
    if pretrained_glove_emb is None:
        word2id_dict = pickle.load(open(os.path.join(os.path.dirname(args.test_data_file), 'word2id_dict.pkl'),'rb'))
    test_data, word2id_dict = dh.load_data_and_labels(
        data_file=args.test_data_file,
        num_classes_list=args.num_classes_list,
        embedding_size=args.embedding_dim,
        word2id_dict=word2id_dict)
    num_samples = len(test_data.labels)

    BEST_OR_LATEST = input("☛ Load Best or Latest Model?(B/L): ")

    while not (BEST_OR_LATEST.isalpha() and BEST_OR_LATEST.upper() in ['B', 'L']):
        BEST_OR_LATEST = input("✘ The format of your input is illegal, please re-input: ")
    dataname = args.dataname
    if BEST_OR_LATEST.upper() == 'B':
        print("✔︎ Loading best model...")
        best_checkpoint_dir = os.path.join('runs', MODEL, 'bestcheckpoints')
        checkpoint_file = cm.get_best_checkpoint(best_checkpoint_dir, select_maximum_value=True)
        log_file = os.path.join('model', '_'.join([dataname, MODEL,'B' ,str(args.beta)]) + '.txt')
    else:
        print("✔︎ Loading latest model...")
        checkpoint_dir = os.path.join('runs', MODEL, 'checkpoints')
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        log_file = os.path.join('model', '_'.join([dataname, MODEL, 'L', str(args.beta)]) + '.txt')
    logger = dh.logger(log_file)
    print(checkpoint_file)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            saver = tf.train.import_meta_graph("{0}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            input_x = graph.get_operation_by_name("input_x").outputs[0]
            input_y_list = [
                graph.get_operation_by_name("input_y_{}".format(i)).outputs[0] for i in
                range(len(args.num_classes_list.split(',')))
            ]
            input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout = graph.get_operation_by_name("dropout").outputs[0]
            beta = graph.get_operation_by_name("beta").outputs[0]
            is_training = graph.get_operation_by_name("is_training").outputs[0]

            local_scores_list = [
                graph.get_operation_by_name("Local_Predict_Layer_{}/scores".format(i)).outputs[0] for i in
                range(len(args.num_classes_list.split(',')))
            ]
            global_scores = graph.get_operation_by_name("global-output/global_scores").outputs[0]
            combine_scores = graph.get_operation_by_name("output/combine_scores").outputs[0]
            loss = graph.get_operation_by_name("loss/loss").outputs[0]

            batches_test = dh.batch_iter_test(
                list(zip(test_data.content_tokenindex, test_data.labels, test_data.labels_tuple)),
                args.batch_size,
                1,
                args.pad_seq_len,
                list(map(int, args.num_classes_list.split(','))),
                sum(list(map(int, args.num_classes_list.split(',')))))

            test_counter, test_loss = 0, 0.0

            num_classes_list = [int(i) for i in args.num_classes_list.split(',')]
            depth_all = len(num_classes_list)
            total_classes = sum(num_classes_list)

            true_onehot_labels = lil_matrix((num_samples, total_classes))
            true_onehot_labels_level_dict = {
                i: lil_matrix((num_samples, num_class)) for i, num_class in enumerate(num_classes_list)
            }
            predicted_onehot_labels_ts_local = lil_matrix((num_samples, total_classes))
            predicted_onehot_labels_ts_local_level = {
                i: lil_matrix((num_samples, num_class)) for i, num_class in enumerate(num_classes_list)}
            predicted_onehot_labels_ts_global = lil_matrix((num_samples, total_classes))
            predicted_onehot_labels_ts_global_level = {
                i: lil_matrix((num_samples, num_class)) for i, num_class in enumerate(num_classes_list)}
            predicted_onehot_labels_ts_combine = lil_matrix((num_samples, total_classes))
            predicted_onehot_labels_ts_combine_level = {
                i: lil_matrix((num_samples, num_class)) for i, num_class in enumerate(num_classes_list)}
            predicted_onehot_labels_tk_global = {
                i: lil_matrix((num_samples, total_classes)) for i in range(args.top_num)}
            predicted_onehot_labels_tk_combine = {
                i: lil_matrix((num_samples, total_classes)) for i in range(args.top_num)}
            predicted_onehot_labels_tk_local = {
                i: lil_matrix((num_samples, total_classes)) for i in range(args.top_num)}
            predicted_global_scores = []
            predicted_global_scores_level = [[] for i in num_classes_list]
            predicted_local_scores = []
            predicted_local_scores_level = [[] for i in num_classes_list]
            predicted_combine_scores = []
            predicted_combine_scores_level = [[] for i in num_classes_list]

            num_sample_id_base = 0
            for x_batch_test, y_batch_test_onehot, y_batch_test_tuple_onehot in batches_test:
                feed_dict_local_y = dict()
                for idx, y_batch_local in enumerate(y_batch_test_tuple_onehot):
                    feed_dict_local_y[input_y_list[idx]] = y_batch_local
                feed_dict = {
                    input_x: x_batch_test,
                    input_y: y_batch_test_onehot,
                    dropout: 1.0,
                    beta: args.beta,
                    is_training: False
                }
                feed_dict.update(feed_dict_local_y)

                results = sess.run(local_scores_list + [global_scores, combine_scores, loss], feed_dict)
                batch_local_scores_list = results[:-3]
                batch_global_scores, batch_combine_scores, cur_loss = results[-3:]
                batch_global_scores_list = \
                    global_score_to_hierarch(batch_global_scores, [int(i) for i in num_classes_list])
                batch_combine_scores_list = \
                    global_score_to_hierarch(batch_combine_scores, [int(i) for i in num_classes_list])
                batch_local_scores = np.hstack(batch_local_scores_list)

                true_onehot_labels[num_sample_id_base: num_sample_id_base + len(y_batch_test_onehot),:] = y_batch_test_onehot
                for level_i in range(depth_all):
                    true_onehot_labels_level_dict[level_i][
                    num_sample_id_base: num_sample_id_base + len(y_batch_test_onehot), :] = \
                        lil_matrix(y_batch_test_tuple_onehot[level_i])

                for i, num_class in enumerate(num_classes_list):
                    predicted_combine_scores_level[i].append(batch_combine_scores_list[i])
                    predicted_global_scores_level[i].append(batch_global_scores_list[i])
                    predicted_local_scores_level[i].append(batch_local_scores_list[i])
                predicted_global_scores.append(batch_global_scores)
                predicted_combine_scores.append(batch_combine_scores)
                predicted_local_scores.append(batch_local_scores)

                batch_predicted_labels_ts_local = \
                    dh.get_label_threshold(scores=batch_local_scores, threshold=args.threshold)
                batch_predicted_labels_ts_global = \
                    dh.get_label_threshold(scores=batch_global_scores, threshold=args.threshold)
                batch_predicted_labels_ts_combine = \
                    dh.get_label_threshold(scores=batch_combine_scores, threshold=args.threshold)
                predicted_onehot_labels_ts_local[
                num_sample_id_base: len(y_batch_test_onehot) + num_sample_id_base, :] = batch_predicted_labels_ts_local
                predicted_onehot_labels_ts_global[
                num_sample_id_base: len(y_batch_test_onehot) + num_sample_id_base, :] = batch_predicted_labels_ts_global
                predicted_onehot_labels_ts_combine[
                num_sample_id_base: len(y_batch_test_onehot) + num_sample_id_base,:] = batch_predicted_labels_ts_combine

                for i, num_class in enumerate(num_classes_list):
                    batch_predicted_labels_ts_local_i = \
                        dh.get_label_threshold(scores=batch_local_scores_list[i], threshold=args.threshold)
                    batch_predicted_labels_ts_global_i = \
                        dh.get_label_threshold(scores=batch_global_scores_list[i], threshold=args.threshold)
                    batch_predicted_labels_ts_combine_i = \
                        dh.get_label_threshold(scores=batch_combine_scores_list[i], threshold=args.threshold)

                    predicted_onehot_labels_ts_local_level[i][
                        num_sample_id_base: len(y_batch_test_onehot) + num_sample_id_base, :] = \
                        batch_predicted_labels_ts_local_i
                    predicted_onehot_labels_ts_global_level[i][
                    num_sample_id_base: len(y_batch_test_onehot) + num_sample_id_base, :] = \
                        batch_predicted_labels_ts_global_i
                    predicted_onehot_labels_ts_combine_level[i][
                    num_sample_id_base: len(y_batch_test_onehot) + num_sample_id_base, :] = \
                        batch_predicted_labels_ts_combine_i

                # Get one-hot prediction by topK
                for i in range(args.top_num):
                    batch_predicted_labels_tk_local_i = dh.get_label_topk(scores=batch_local_scores, top_num=i + 1)
                    batch_predicted_labels_tk_global_i = dh.get_label_topk(scores=batch_global_scores, top_num=i + 1)
                    batch_predicted_labels_tk_combine_i = dh.get_label_topk(scores=batch_combine_scores, top_num=i + 1)
                    predicted_onehot_labels_tk_local[i][num_sample_id_base: len(y_batch_test_onehot) + num_sample_id_base,:] =\
                        batch_predicted_labels_tk_local_i
                    predicted_onehot_labels_tk_global[i][num_sample_id_base: len(y_batch_test_onehot) + num_sample_id_base, :] = \
                        batch_predicted_labels_tk_global_i
                    predicted_onehot_labels_tk_combine[i][num_sample_id_base: len(y_batch_test_onehot) + num_sample_id_base, :] = \
                        batch_predicted_labels_tk_combine_i


                test_loss = test_loss + cur_loss
                test_counter = test_counter + 1
                num_sample_id_base += len(y_batch_test_onehot)

            pre_lsit, rec_list, F_list, prc_list, auc_list, \
            pre_list_level, rec_list_level, F_list_level, prc_list_level, auc_list_level, \
            pre_tk, rec_tk, F_tk = evaluate(
                true_onehot_labels = true_onehot_labels,
                true_onehot_labels_level_dict = true_onehot_labels_level_dict,
                predicted_onehot_labels_ts = predicted_onehot_labels_ts_global,
                predicted_onehot_labels_ts_level = predicted_onehot_labels_ts_global_level,
                predicted_onehot_labels_tk = predicted_onehot_labels_tk_global,
                predict_scores = predicted_global_scores,
                predict_scores_level = predicted_global_scores_level,
                num_classes_list = num_classes_list,
                emb_type='global',logger=logger)
            pre_lsit, rec_list, F_list, prc_list, auc_list, \
            pre_list_level, rec_list_level, F_list_level, prc_list_level, auc_list_level, \
            pre_tk, rec_tk, F_tk = evaluate(
                true_onehot_labels=true_onehot_labels,
                true_onehot_labels_level_dict=true_onehot_labels_level_dict,
                predicted_onehot_labels_ts=predicted_onehot_labels_ts_local,
                predicted_onehot_labels_ts_level=predicted_onehot_labels_ts_local_level,
                predicted_onehot_labels_tk=predicted_onehot_labels_tk_local,
                predict_scores=predicted_local_scores,
                predict_scores_level=predicted_local_scores_level,
                num_classes_list=num_classes_list,
                emb_type='local',logger=logger)
            pre_lsit, rec_list, F_list, prc_list, auc_list, \
            pre_list_level, rec_list_level, F_list_level, prc_list_level, auc_list_level, \
            pre_tk, rec_tk, F_tk = evaluate(
                true_onehot_labels=true_onehot_labels,
                true_onehot_labels_level_dict=true_onehot_labels_level_dict,
                predicted_onehot_labels_ts=predicted_onehot_labels_ts_combine,
                predicted_onehot_labels_ts_level=predicted_onehot_labels_ts_combine_level,
                predicted_onehot_labels_tk=predicted_onehot_labels_tk_combine,
                predict_scores=predicted_combine_scores,
                predict_scores_level=predicted_combine_scores_level,
                num_classes_list=num_classes_list,
                emb_type='combine',logger=logger)

    print("✔︎ Done.")


if __name__ == '__main__':
    args = parse_args()
    test_lahcn(args)
