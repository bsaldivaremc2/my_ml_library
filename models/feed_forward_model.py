import tensorflow as tf
from datetime import datetime
import os

def dice_coef(y_true, y_pred):
    tp_raw = tf.math.multiply(y_true,y_pred)
    tp = tf.math.reduce_sum(tp_raw)
    #
    neg_y_pred = (y_pred - 1)*-1
    neg_y_true = (y_true - 1)*-1
    #
    fp_raw = tf.math.multiply(neg_y_true,y_pred)
    fp = tf.math.reduce_sum(fp_raw)
    #
    fn_raw = tf.math.multiply(y_true,neg_y_pred)
    fn = tf.math.reduce_sum(fn_raw)
    #
    dice = (2*tp)/(2*tp+fp+fn)
    return dice

#2 TP / (2 TP + FP + FN)
class CustomDiceLoss(tf.keras.losses.Loss):
    def __init__(self,smooth=100):
        super().__init__()
        self.smooth=smooth
    def call(self, y_true, y_pred):
        tp_raw = tf.math.multiply(y_true,y_pred)
        tp = tf.math.reduce_sum(tp_raw)
        #
        neg_y_pred = (y_pred - 1)*-1
        neg_y_true = (y_true - 1)*-1
        #
        fp_raw = tf.math.multiply(neg_y_true,y_pred)
        fp = tf.math.reduce_sum(fp_raw)
        #
        fn_raw = tf.math.multiply(y_true,neg_y_pred)
        fn = tf.math.reduce_sum(fn_raw)
        #
        dice = (2*tp)/(2*tp+fp+fn)
        return 1 - dice

def get_model(ishape,feat_names=[],ns= [10,10,10],compile_model=True,learning_rate=0.01,
             input_shape=[300],output_shape=[1],
              output_activation="sigmoid",
             activations="relu",
             loss="binary_classification",metrics="dice_coef"):
    
    #from tensorflow import keras
    #from tensorflow.keras import layers
    #tf.keras.losses.MeanSquaredError()
    if len(feat_names)>0:
        inputs = { fn:tf.keras.Input(shape=(1,)) for fn in feat_names }
        x = tf.keras.layers.Concatenate()(inputs.values())
    else:
        inputs = tf.keras.Input(shape=ishape)
        x = inputs
    for n in ns:
        if activations=="relu":
            x = tf.keras.layers.Dense(n, activation="relu")(x)
        elif activations=="leaky_relu":
            x = tf.keras.layers.Dense(n, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
    if output_shape==[1]:
        if output_activation == "sigmoid":
            x = tf.keras.layers.Dense(1, activation=output_activation)(x)
        elif output_activation=="softmax":
            pass # to implement
    else:
        pass # to implement
    om = tf.keras.Model(inputs=inputs, outputs=x)
    #Loss
    if loss=="mse":
        loss = tf.keras.losses.MeanSquaredError(reduction="sum", name="mean_squared_error")
    elif loss=="binary_classification":
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    elif loss=="dice_coef":
        loss=CustomDiceLoss()
    #Metrics
    if metrics=="dice_coef":
        metrics=[dice_coef]
    if compile_model:
        om.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=loss,metrics=metrics)
    return om

def run_train(train_ds,val_ds,
              feat_names=[],
              epochs=10,ns=[10,10,10,10,10],
              learning_rate=0.01,
              input_shape=[300],output_shape=[1],
              output_activation="sigmoid",
              loss="binary_classification",
              metrics="dice_coef",
              imodel="",
             checkpoint_dir="./checkpoints/"):
    DATA_DIR = checkpoint_dir
    EPOCHS = epochs
    STEPS_PER_EPOCH=None
    VALIDATION_STEPS=None
    if type(imodel)==str:
        model = get_model(input_shape,
                          feat_names=feat_names,
                          ns=ns,
                          compile_model=True,learning_rate=learning_rate,
                          output_shape=[1],
                          output_activation=output_activation,
                          loss=loss
                         )
    else:
        model = imodel
    checkpoint_filepath = '{}checkpoints/'.format(DATA_DIR)
    os.makedirs(checkpoint_filepath,exist_ok=True)
    #
    time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = "{}logs/scalars/{}/".format(DATA_DIR,time_now)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    #
    class haltCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('val_loss') > 0.8):
                print("\n\n\nReached >=0.7 val loss value so cancelling training!\n\n\n")
                self.model.stop_training = True
    trainingStopCallback = haltCallback()
    #
    early = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    #
    if metrics=="dice_coef":
        mode="max"
        monitor="val_dice_coef"
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor=monitor,
        mode=mode,
        save_best_only=True)
    """
    if isinstance(trainx,pd.DataFrame) or isinstance(trainx,np.array):
        model_history = model.fit(trainx,trainy, epochs=EPOCHS,validation_split=0.5,
                        callbacks=[model_checkpoint_callback])
    """
    model_history = model.fit(train_ds,epochs=EPOCHS,validation_data=val_ds,
                        callbacks=[model_checkpoint_callback,tensorboard_callback,
                                  trainingStopCallback,early])
    model.load_weights(checkpoint_filepath)
    return model,model_history
