## Setting up evaluation metrics used to assess the classifier
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

## Setting up IOU (intersection over union) metrics
def IOU(y_true,y_pred,y_bb,y_pred_bb):
    y_true = np.reshape(y_true,(-1,1))
    y_pred = np.reshape(y_pred,(-1,1))
    y_bb = np.reshape(y_bb,(-1,4))
    y_pred_bb = np.reshape(y_pred_bb,(-1,4))
    mask = (y_true==1)
    iou = []

    for i, (t, p) in enumerate(zip(y_bb,y_pred_bb)):
        ## convert width and height back to linear values
        tx, ty, tw, th = t
        px, py, pw, ph = p

        tw = tw**2
        th = th**2
        pw = pw**2
        ph = ph**2

        ## If there is no object present, IOU = -1 as indicator
        if not mask[i]:
            iou.append(-1)
            continue

        ## If bounding boxes do not intersect, IOU = 0
        case_1 = ((tx-tw/2)>=(px+pw/2))or((ty-th/2)>=(py+ph/2))
        case_2 = ((tx+tw/2)<=(px-pw/2))or((ty+th/2)<=(py-ph/2))
        if case_1 or case_2:
            iou.append(0)
            continue

        ## Calculate intersection
        bb_tlX = max(tx-tw/2,px-pw/2)
        bb_tlY = max(ty-th/2,py-ph/2)
        bb_brX = min(tx+tw/2,px+pw/2)
        bb_brY = min(ty+th/2,py+ph/2)
        assert bb_brX>bb_tlX, \
            "X value of bounding box invalid"
        assert bb_brY>bb_tlY, \
            "Y value of bounding box invalid"
        intsec = (bb_brX-bb_tlX)*(bb_brY-bb_tlY)
        assert intsec > 0, \
            "Invalid value encountered when calculating intersection"

        ## Calculate union
        union = (tw*th)+(pw*ph)-intsec
        assert union > 0, \
            "Invalid value encountered when calculating union"
        iou.append(intsec/union)

    assert len(iou) == len(y_true), \
        "IOU results incomplete"
    return np.array(iou)

def metrics(y_true,y_pred_proba,y_bb,y_pred_bb):
    y_pred = np.argmax(y_pred_proba,axis=-1)
    iou = IOU(y_true,y_pred,y_bb,y_pred_bb)
    mean_iou = np.mean(iou[iou>=0])
    cm = confusion_matrix(y_true,y_pred)
    #true_negative = cm[0,0]
    false_positive = cm[0,1]
    false_negative = cm[1,0]
    true_positive = cm[1,1]
    recall = true_positive/(true_positive+false_negative)
    precision = true_positive/(true_positive+false_positive)
    accuracy = sum(y_true == y_pred)/y_true.shape[0]
    auc = roc_auc_score(y_true,y_pred_proba[:,1])
    print(
        "IOU: {0:.2f},\
        Accuracy: {1:.2f}, \
        AUC:{2:.2f}, \
        Precision: {3:.2f}, \
        Recall: {4:.2f}".format(mean_iou,accuracy,auc,precision,recall))

if __name__ == "__main__":
    pass
