import pandas as pd

train_files = ["../data/{}/CEdata_train_pairs.csv".format(x) for x in ["CEtrain", "SUP1data_text", "SUP2data_text", "SUP3data_text"]]
train_files += ["../data/CEfinal/CEfinal_train_pairs.csv"]

new_train = pd.concat([pd.read_csv(x) for x in train_files])
new_train.to_csv("../data/combination/pairs.csv", index=False)

publicinfo_files = [x.replace("pairs", "publicinfo") for x in train_files]
new_publicinfo = pd.concat([pd.read_csv(x) for x in publicinfo_files])
new_publicinfo.to_csv("../data/combination/publicinfo.csv", index=False)

target_files = [x.replace("pairs", "target") for x in train_files]
new_target = pd.concat([pd.read_csv(x, names=["SampleID", "Target", "Details"], skiprows=1) for x in target_files])
new_target.to_csv("../data/combination/target.csv", index=False)
