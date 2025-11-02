import glob, os
from sklearn.model_selection import KFold
import pandas as pd

bags_path = glob.glob('temp_train/*.pt')
bags_path = sorted(bags_path)  # ensure consistent order

kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, test_idx) in enumerate(kf.split(bags_path)):
    train_files = [os.path.basename(bags_path[i]) for i in train_idx]
    test_files = [os.path.basename(bags_path[i]) for i in test_idx]
    df_train = pd.DataFrame(train_files, columns=['slide'])
    df_test = pd.DataFrame(test_files, columns=['slide'])
    df_train.to_csv(f"results/fold_{fold}_train.csv", index=False)
    df_test.to_csv(f"results/fold_{fold}_test.csv", index=False)
    print(f"Saved fold {fold} splits with {len(train_files)} train and {len(test_files)} test slides.")
