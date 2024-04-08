# accuracy
#
# What percent of classifications are correct?
# 
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
# return: percent accuracy bounded between [0, 1]
#
def accuracy(true, pred):
    acc = None
    num_correct = 0
    for x in range(0, len(true)):
        if pred[x] == true[x]:
            num_correct += 1
    acc = num_correct / len(pred)
    return acc

# binary_f1 
#
# A method to calculate F-1 scores for a binary classification task.
# 
# args -
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
# selected_class: Boolean - the selected class the F-1 
#                 is being calculated for.
# 
# return: F-1 score between [0, 1]
#
"""
TP_1 1==1
FP_1 0==1
TN_1 0==0
FN_1 1==0

TP_0 0==0
FP_0 1==0
TN_0 1==1
FN_0 0==1
"""
def binary_f1(true, pred, selected_class=True):
    f1 = None
    precision = 0
    recall = 0
    
    true_pos, false_neg, false_pos = (0, 0, 0)

    # Compute true positive
    for i in range(0, len(pred)):
        if selected_class:
            if true[i] == 1 and true[i] == pred[i]:
                true_pos += 1
        else:
            if true[i] == 0 and true[i] == pred[i]:
                true_pos += 1
    # Compute false negative
    for i in range(0, len(pred)):
        if selected_class:
            if pred[i] == 0 and true[i] == 1:
                false_neg += 1
        else:
            if pred[i] == 1 and true[i] == 0:
                false_neg += 1
    # Compute false positive
    for i in range(0, len(pred)):
        if selected_class:
            if true[i] == 0 and pred[i] == 1:
                false_pos += 1
        else:
            if true[i] == 1 and pred[i] == 0:
                false_pos += 1    
    if true_pos + false_pos != 0:
        precision = true_pos / (true_pos + false_pos)
    else: 
        precision = 0
    if true_pos + false_neg != 0:
        recall = true_pos / (true_pos + false_neg)
    else:
        recall = 0
    if precision + recall != 0:
        f1 = (2*precision*recall) / (precision+recall)
    else:
        f1 = 0
    return f1

# binary_macro_f1
# 
# Averaged F-1 for all selected (true/false) classes.
#
# args -
# true: ground truth, Python list of booleans.
# pred: model predictions, Python list of booleans.
#
#
def binary_macro_f1(true, pred):
    averaged_macro_f1 = None
    averaged_macro_f1 = (binary_f1(true, pred, selected_class=False) + binary_f1(true, pred, selected_class=True)) / 2
    return averaged_macro_f1