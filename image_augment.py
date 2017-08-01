import numpy as np

def rand_brightness(data,max_factor):
    #maxfactor: maximum scale of adjustment(from 0-1)
    ##Randomly adjust brightness of each image separately
    adj_arr = np.random.uniform(1-max_factor,1+max_factor,size=(data.shape[0]))
    adj_data = np.clip(data*adj_arr[:,np.newaxis,np.newaxis,np.newaxis],-1.0,1.0)
    return adj_data

def rand_contrast(data,max_factor):
    ##Randomly adjust contrast of each image separately
    max_factor = max_factor*255
    adj_arr = np.random.uniform(-max_factor,+max_factor,size=(data.shape[0]))
    adj_arr = (259*(adj_arr+255))/(255*(259-adj_arr))
    adj_data = np.clip((adj_arr[:,np.newaxis,np.newaxis,np.newaxis]*(data-1.0))+1.0,-1.0,1.0)
    return adj_data

def augment(data,max_factor,rand_order=False):
    ##Brightness and contrast adjustment are not commutative
    ##Augment with random order of brightness/contrast adjustment
    if rand_order:
        i = np.random.randint(0,1)
        if i == 1:
            adj_data = rand_contrast(data,max_factor)
            adj_data = rand_brightness(adj_data,max_factor)
            return adj_data

    adj_data = rand_brightness(data,max_factor)
    adj_data = rand_contrast(adj_data,max_factor)
    return adj_data

def sq_center_crop(img):
    width = img.size[0]
    height = img.size[1]
    size = min(width,height)
    w_adj = (width-size)/2
    h_adj = (height-size)/2
    img_crop = img.crop((w_adj,h_adj,width-w_adj,height-h_adj))
    return img_crop