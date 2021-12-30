


from PIL import Image
import numpy as np
import torch

from tools import common
from tools.dataloader import norm_RGB
from nets.patchnet import *



import os

import numpy as np
import cv2
import torch


import matplotlib.pyplot as plt

class Frame:
    def __init__(self, id, img, kps, desc ):
        self.id = id
        self.image = img
        self.keypoints = kps
        self.descriptors = desc
  
    def getitems(self):
        return (self.image, self.keypoints, self.descriptors, self.filename)

    def get_kp_desc(self):
        return (self.keypoints, self.descriptors)

    def get_image(self):
        return self.image

    def get_file(self):
        return self.filename

class FramePair:
    def __init__(self, f1, f2, matches_no, left_kp, right_kp ):
        self.frame1 = f1
        self.frame2 = f2
        self.left_kp = left_kp
        self.right_kp = right_kp



        self.matches_no = matches_no

    def getkeypts(self):
        return (self.left_kp, self.right_kp)
    
       
class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]



def ratio_mutual_nn_matcher(descriptors1, descriptors2, ratio=0.90):
    # Lowe's ratio matcher + mutual NN for L2 normalized descriptors.
    

    device = descriptors1.device
    sim = descriptors1 @ descriptors2.t()
    nns_sim, nns = torch.topk(sim, 2, dim=1)
    nn12 = nns[:, 0]
    nns_dist = torch.sqrt(2 - 2 * nns_sim)
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    matches = torch.stack([ids1, nns[:, 0]])
    ratios = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
    mask = torch.min(ids1 == nn21[nn12], ratios <= ratio)
    matches = matches[:, mask]
    return matches.t().data.cpu().numpy(), nns_dist[mask, 0]

def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()





def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                        min_scale=0.0, max_scale=1, 
                        min_size=256, max_size=1280, 
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
                
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = detector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
        s /= scale_f

        # down-scale the image for next iteration
        break
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores


def extract_keypoints(img, args):
    

        
    xys, desc, scores = extract_multiscale(net, img, detector,
        scale_f   = args['scale_f'], 
        min_scale = args['min_scale'], 
        max_scale = args['max_scale'],
        min_size  = args['min_size'], 
        max_size  = args['max_size'], 
        verbose = False)

    xys = xys.cpu().numpy()
    scores = scores.cpu().numpy()
    idxs = np.argwhere(scores>0.85)

    return (xys[idxs], desc[idxs])
    

args = {'model' : './models/r2d2_WAF_N16.pt', 'scale_f' : 2**0.25, 'min_size' : 256, 'max_size' : 1380, 'min_scale' : 0, 'max_scale' : 1, 'reliability_thr' : 0.7, 'repeatability_thr' : 0.7 , 'gpu' : [0]}
net = load_network(args['model'])
iscuda = common.torch_set_gpu(args['gpu'])
if iscuda: net = net.cuda()
detector = NonMaxSuppression( rel_thr = args['reliability_thr'], rep_thr = args['repeatability_thr'])


def extract_features_and_desc(image):
    '''
    image: np.uint8
    '''
    img_pil = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_pil)
    img_cpu = img_pil
    # print("type(img_cpu):",type(img_cpu))
    img = norm_RGB(img_cpu)[None]
    if iscuda: 
      img = img.cuda()
    kps, desc = extract_keypoints(img, args)
    # alldesc = np.transpose(alldesc, (1, 2,0))

    return np.squeeze(kps), np.squeeze(desc)

def get_matches(ref_kp, ref_desc, cur_kp, cur_desc):
    matches = ratio_mutual_nn_matcher(ref_desc, cur_desc)[0]

    return matches


def vizualize_matches(framepair):
    cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in framepair.left_kp]
    cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in framepair.right_kp]
    dmtches = [cv2.DMatch(_imgIdx=0,_queryIdx=i, _trainIdx=i, _distance = 0) for i in range(len(cv_kp1))]
    out_img = cv2.drawMatches(framepair.frame1.image, cv_kp1, framepair.frame2.image, cv_kp2, dmtches[::50], None, (0,255,255), -1, None, 2)
    out_img=cv2.cvtColor(out_img,cv2.COLOR_BGR2RGB)
    plt.imshow(out_img)
    plt.show()
    



# ref_img=cv2.imread('./imgs/brooklyn.png')
# cur_img=ref_img.copy()
# num_rows, num_cols = ref_img.shape[:2]
# rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 10, 1)
# cur_img = cv2.warpAffine(ref_img, rotation_matrix, (num_cols, num_rows))


# imgidx = 1



# ref_kp, ref_desc = extract_features_and_desc(ref_img) 

# frame1 = Frame(id = 0, img = ref_img, kps = ref_kp, desc = ref_desc )


# cur_kp, cur_desc = extract_features_and_desc(cur_img)


# frame2 = Frame(id = imgidx, img = cur_img, kps = cur_kp, desc = cur_desc)

# avg_keypts=(len(ref_kp)+len(cur_kp))/2

# matches = get_matches(ref_kp, ref_desc, cur_kp, cur_desc)
# number_matches=len(matches)

# ref_keypoints = ref_kp[matches[:, 0], : 2].astype(np.float32)
# cur_keypoints = cur_kp[matches[:, 1], : 2].astype(np.float32)



# framepair = FramePair(frame1, frame2, matches, ref_keypoints, cur_keypoints)

# print("METRIC " ,number_matches/avg_keypts)
# vizualize_matches(framepair)

images=os.listdir('./data_jan_19/HD')
img=cv2.imread('./data_jan_19/HD/'+images[0])
ref_img=img[:,:img.shape[1]//2,:]
cur_img=img[:,img.shape[1]//2:,:]
cv2.imshow("IMG",cur_img)
cv2.imshow("REF",ref_img)
cv2.waitKey(0)