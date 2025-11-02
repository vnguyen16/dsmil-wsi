# test_features_from_csv.py
import torch, os, numpy as np, pandas as pd
import torch.nn as nn
from collections import OrderedDict
import dsmil as mil
import glob

# --- config ---
weights_path = "weights/20251020/fold_4_1.pth"
bag_dir = "temp_train"  # directory containing .pt feature bags
test_csv = r"C:\Users\Vivian\Documents\PANTHER\PANTHER\src\splits\cross-val\FA_PT_k=0\test.csv"  # CSV listing test slides
num_classes = 1
feats_size = 2048  # must match training args
output_csv = "results/cv/test_predictions_fold4.csv"

# --- load model ---
i_classifier = mil.FCLayer(in_size=feats_size, out_size=num_classes).cuda()
b_classifier = mil.BClassifier(input_size=feats_size, output_class=num_classes).cuda()
milnet = mil.MILNet(i_classifier, b_classifier).cuda()
milnet.load_state_dict(torch.load(weights_path))
milnet.eval()

# --- load test slide list ---
df_test = pd.read_csv(test_csv)

# Handle common column name patterns
if "slide" in df_test.columns:
    test_slides = df_test["slide"].tolist()
elif "slide_id" in df_test.columns:
    test_slides = df_test["slide_id"].tolist()
else:
    test_slides = df_test.iloc[:, 0].tolist()  # fallback if unnamed column

# Normalize slide names (remove extensions if present)
test_slides = [os.path.splitext(s)[0] for s in test_slides]

print(f"Found {len(test_slides)} test slides in {test_csv}")

# --- test loop ---
results = []
for slide_name in test_slides:
    # Match bag file by name
    pattern = os.path.join(bag_dir, f"{slide_name}*.pt")
    matches = glob.glob(pattern)
    if not matches:
        print(f"Warning: no .pt file found for slide {slide_name}")
        continue

    fpath = matches[0]
    bag = torch.load(fpath)
    feats = bag[:, :feats_size].cuda()
    label = bag[0, feats_size:].cuda()

    with torch.no_grad():
        ins_pred, bag_pred, _, _ = milnet(feats)
        prob = torch.sigmoid(bag_pred).squeeze().cpu().numpy()
        pred = (prob > 0.5).astype(int)

    results.append({
        "slide_id": slide_name,
        "true_label": float(label.cpu().numpy()[0]),
        "pred_label": int(pred),
        "prob": float(prob),
    })

    print(f"{slide_name} → Pred: {pred}, Prob: {prob:.4f}")

# --- save results ---
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\nSaved results to {output_csv}")
