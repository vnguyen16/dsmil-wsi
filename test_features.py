# test_features.py
import torch, glob, os, numpy as np
import torch.nn as nn
from collections import OrderedDict
import dsmil as mil

# --- config ---
weights_path = "weights/20251020/fold_0_1.pth"
bag_dir = "temp_train"  # or wherever your .pt feature bags are
num_classes = 1
feats_size = 2048  # must match training args

# --- load model ---
i_classifier = mil.FCLayer(in_size=feats_size, out_size=num_classes).cuda()
b_classifier = mil.BClassifier(input_size=feats_size, output_class=num_classes).cuda()
milnet = mil.MILNet(i_classifier, b_classifier).cuda()
milnet.load_state_dict(torch.load(weights_path))
milnet.eval()

# --- test loop ---
bag_files = glob.glob(os.path.join(bag_dir, "*.pt"))
for fpath in bag_files:
    bag = torch.load(fpath)
    feats = bag[:, :feats_size].cuda()
    label = bag[0, feats_size:].cuda()
    with torch.no_grad():
        ins_pred, bag_pred, _, _ = milnet(feats)
        prob = torch.sigmoid(bag_pred).squeeze().cpu().numpy()
        pred = (prob > 0.5).astype(int)
    print(f"{os.path.basename(fpath)} → Pred: {pred}, Prob: {prob}")
