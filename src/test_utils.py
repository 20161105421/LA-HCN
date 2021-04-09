import numpy as np

def build_hierarch_labels(y_batch_test_tuple, depth_all):
    y_batch_test_level = dict()
    for hierarch_labels_i in y_batch_test_tuple:
        for level_i in range(depth_all):
            if level_i not in y_batch_test_level:
                y_batch_test_level[level_i] = []
            y_batch_test_level[level_i].append(hierarch_labels_i[level_i])
    return y_batch_test_level

def global_score_to_hierarch(score,hierarch_structure = [8,110,431,1473]):
    # batch_first_scores, batch_second_scores, batch_third_scores, batch_fourth_scores = global_score_to_hierarch(batch_scores)
    start = 0
    output = []
    for num_class in hierarch_structure:
        end = start + num_class
        batch_level_score = score[:,start:end]
        start = end
        output.append(batch_level_score)
    return output


def average_hit_score(y_true,y_pred):
    hit_score = np.mean(np.any(np.logical_and(y_true, y_pred), 1))
    return hit_score