
EPOCHS = 128
TEST_SPLIT = 0.25
VAL_SPLIT = 0.25
BATCH_SIZE = 2048
LEARNING_RATE = 0.00001
NET = "16384,8192,4096,2048,1024,512,256,128,64,32,16,8,4,2"
D = "/home/bsaldivar/ml/bbv/datasets/joined_data/"
DATA_DIR = "./model_data/"

from  feed_forward_model import *
from tf_make_dataset import *
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
DATA_DIR = "{}{}/{}/".format(DATA_DIR,NET.replace(",","_"),time_now)

def get_test(idf,test_split=0.25):
    ns = list(idf.index)
    n = len(ns)
    nt = int(n*test_split)
    tsi = np.random.choice(ns,nt,replace=False)
    tri = [_ for _ in ns if _ not in tsi]
    return idf.loc[tri],idf.loc[tsi]


#tr = pd.read_csv(D+"train_train_imp_no_id.csv")
tr = pd.read_csv(D+"train_train_imp_no_id.csv")
val = pd.read_csv(D+"train_val_imp_no_id.csv")
ts = pd.read_csv(D+"train_test_imp_no_id.csv")
num_feats = tr.shape[1]-1
feat_names = list(tr.columns)[1:]

tr['target'] = tr['target'].astype("float32")
val['target'] = val['target'].astype("float32")
ts['target'] = ts['target'].astype("float32")


print("Proc tf datasets")
trainds = df_to_dataset(tr, shuffle=True, batch_size=BATCH_SIZE,buffer_size=2048)
valds = df_to_dataset(val, shuffle=False, batch_size=BATCH_SIZE,buffer_size=2048)
testds = df_to_dataset(ts, shuffle=False, batch_size=BATCH_SIZE,buffer_size=2048)

ns = [int(_.strip()) for _ in NET.split(",") if len(_)>0]

model, history = run_train(trainds,valds,feat_names=feat_names,
          epochs=EPOCHS,ns=ns,
              learning_rate=LEARNING_RATE,
              input_shape=[num_feats],output_shape=[1],
              output_activation="sigmoid",
              loss="dice_coef",
              metrics="dice_coef",
              imodel="",
             checkpoint_dir=DATA_DIR)

print("Evaluating test")
val_loss, val_metric = model.evaluate(testds)
val_metric = round(val_metric,6)
model_export_path = "{}models/val_dice_coef_{}/".format(DATA_DIR,val_metric)
os.makedirs(model_export_path,exist_ok=True)
tf.saved_model.save(model, model_export_path)
print("saved model",model_export_path)
