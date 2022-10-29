import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
from  feed_forward_model import *
from tf_make_dataset import *
from datetime import datetime
import numpy as np

#MODEL_DIR = "/home/bsaldivar/ml/bbv/datasets/joined_data/model_data/4096_4096_2048_2048_1024_1024_512_512_256_256_128_128/20221028-185320/models/val_dice_coef_0.486421/"
MODEL_DIR = "/home/bsaldivar/ml/bbv/datasets/joined_data/model_data/models/val_dice_coef_0.487326/"
PRED_DIR = "/home/bsaldivar/ml/bbv/datasets/joined_data/predictions/"
pred_df = "/home/bsaldivar/ml/bbv/datasets/joined_data/test_all_imputed.csv"
os.makedirs(PRED_DIR,exist_ok=True)
#model = tf.saved_model.load(MODEL_DIR)
print("loading model")
model = tf.keras.models.load_model(MODEL_DIR,custom_objects={'dice_coef':dice_coef,'CustomDiceLoss':CustomDiceLoss},
                                  compile=False)
print("loading test set for predictions")
df = pd.read_csv(pred_df,index_col=0)
df['target']=np.zeros((df.shape[0],1))
df_for_pred = df_to_dataset(df, shuffle=False, batch_size=2048)
print("Predicting")
preds = model.predict(df_for_pred)
_ = pd.DataFrame()
_['ID'] = [ i.split("-")[0] for i in df.index.values]
_['target']=preds.round().astype(int)
time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
of = "{}prediction_{}.csv".format(PRED_DIR,time_now)
print("saved predictions on",of)
_.to_csv(of,index=False)
