import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from local_metrics import IOU

## Function to convert zero-centered values back to RGB values
def zero_to_rgb(data):
    data = (data*(255/2)+(255/2)).astype(np.uint8)
    return data

#Function to assist in plotting bounding boxes and labels
def plot_traffic(data, p_bb, p_label=[], cols=3, \
                plot_gt=False, gt_bb=None, gt_label=None):
    #Code to to configure display of images in a x by cols grid
    if data.shape[0] < cols:
        cols = data.shape[0]
    if data.shape[0]%cols == 0:
        rows = data.shape[0]//cols
    else:
        rows = (data.shape[0]//cols)+1
    fig, axarr = plt.subplots(nrows=rows, ncols=cols,\
                            sharex=True, sharey=True, figsize=(18,rows*6))
    axarr = np.reshape(axarr,-1)

    #Code to display images and draw bounding boxes
    for i, d in enumerate(data):
        axarr[i].imshow(zero_to_rgb(d))
        im_w = d.shape[0]
        im_h = d.shape[1]
        x = p_bb[i,0]*im_w
        y = p_bb[i,1]*im_h
        w = int(np.square(p_bb[i,2])*im_w)
        h = int(np.square(p_bb[i,3])*im_h)
        tlX = int(x-(w/2))
        tlY = int(y-(h/2))
        annotation = ""
        if len(p_label) > 0:
            annotation += "Vehicle Present: {}".format(p_label[i]==1)

        rect = patches.Rectangle((tlX,tlY), w, h, linewidth=1,\
                                edgecolor='r', facecolor='none')
        axarr[i].add_patch(rect)
        if plot_gt:
            iou = IOU(gt_label[i],p_label[i],gt_bb[i], p_bb[i])[0]
            if iou >= 0:
                annotation = annotation +", IOU: {0:.2f} ".format(iou)
            axarr[i].text(0.05,0.95,annotation,fontsize=12,\
                        color="white", transform=axarr[i].transAxes)
            gt_x = gt_bb[i,0]*im_w
            gt_y = gt_bb[i,1]*im_h
            gt_w = int(np.square(gt_bb[i,2])*im_w)
            gt_h = int(np.square(gt_bb[i,3])*im_h)
            gt_tlX = int(gt_x-(gt_w/2))
            gt_tlY = int(gt_y-(gt_h/2))
            rect_t = patches.Rectangle((gt_tlX,gt_tlY),gt_w,gt_h,linewidth=1,\
                                        edgecolor='b',facecolor='none')
            axarr[i].add_patch(rect_t)
    plt.show()

if __name__ == "__main__":
    pass
