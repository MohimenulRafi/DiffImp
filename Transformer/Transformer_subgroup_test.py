import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, accuracy_score
import os
from metrics import print_metrics_binary
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Masking
import numpy as np
import pandas as pd
import os
import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from metrics import print_metrics_binary
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.isotonic import IsotonicRegression as IR


def Select_Threshold_calibrated(df):
    full_threshold_list = []
    for threshold in np.arange(0.01, 1.0, 0.01):
        #df.drop(columns = ['y_pred'])
        df['y_pred'] = df['score y calibrated'].apply(lambda x: 1 if x >= threshold else 0)

        # survival => 1, death => 0 where 1 minority in this case
        y_pred = df["y_pred"].values
        y_true = df["true y"].values
        f1_C1 = f1_score(y_true, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
        
        full_threshold_list.append([threshold, f1_C1, balanced_accuracy]) # use minority class 1 (death) for LCS and ICU MR
        
    df_varying_threshold = pd.DataFrame(full_threshold_list, columns = ['threshold', 'f1_score', 'balanced_accuracy'])
    
    # select three highest F1 score and the the highest balanced accuracy
    f1_scores = df_varying_threshold["f1_score"].values
    thresholds = df_varying_threshold["threshold"].values
    bal_acc_values = list(df_varying_threshold["balanced_accuracy"].values)
    
    #print(heapq.nlargest(3, f1_scores))
    list_index = heapq.nlargest(3, range(len(f1_scores)), key=f1_scores.__getitem__)
    opt_threshold = thresholds[bal_acc_values.index(max(bal_acc_values[list_index[0]], bal_acc_values[list_index[1]], bal_acc_values[list_index[2]]))]
    
    
    return opt_threshold, df_varying_threshold


# Function to build the Transformer Model
def build_model(input_dim=76, units=64, dropout=0.3, num_classes=1, num_heads=4, num_transformer_blocks=3, ff_dim=16, batch_norm=True):

    inputs = Input(shape=(None, input_dim), name="X")

    # Masking layer for padded sequences
    x = Masking()(inputs)
    
    # Positional encoding layer
    position_embedding = layers.Embedding(input_dim=50, output_dim=input_dim)(tf.range(start=0, limit=48))
    x = x + position_embedding

    # Transformer Encoder blocks
    for _ in range(num_transformer_blocks):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_dim, dropout=dropout)(x1, x1)
        x2 = layers.Add()([x, attention_output])  # Residual connection

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ff_output = layers.Dense(ff_dim, activation="relu")(x3)
        ff_output = layers.Dense(input_dim)(ff_output)
        x = layers.Add()([x2, ff_output])  # Residual connection

    # Global average pooling for sequence output
    x = layers.GlobalAveragePooling1D()(x)

    # Batch normalization
    x = BatchNormalization()(x) if batch_norm else x
    
    # Dropout layer
    x = Dropout(dropout)(x)
    
    # Dense layer for binary classification
    x = Dense(units, activation='relu')(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    # Build and compile the model
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy', tf.keras.metrics.AUC(name="auroc")])
    
    return model




################# Set path #################

# use the test for original test set
def test_model(model, train_X, train_y, train_names,
                        val_X, val_y, val_names,
                        test_X, test_y, test_names,
                        out_dir="", name="", epoch = ""):
    
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    # Step 1: model calibration and threshold selection

    train_prediction = model.predict(train_X)[:, 0]
    val_prediction = model.predict(val_X)[:, 0]
    test_prediction = model.predict(test_X)[:, 0]

    
    ir = IR(out_of_bounds='clip')
    ir.fit( val_prediction, val_y )

    val_prediction_calibrated = ir.transform( val_prediction )
    valid_preds_df = pd.DataFrame()
    valid_preds_df['true y'] = val_y
    valid_preds_df['score y'] = val_prediction
    valid_preds_df['score y calibrated'] = val_prediction_calibrated
    #valid_preds_df.to_csv(os.path.join(output_dir, 'val_preds.csv'), index = False)

    train_prediction_calibrated = ir.transform( train_prediction )
    train_preds_df = pd.DataFrame()
    train_preds_df['true y'] = train_y
    train_preds_df['score y'] = train_prediction
    train_preds_df['score y calibrated'] = train_prediction_calibrated
    #train_preds_df.to_csv(os.path.join(output_dir, 'train_preds.csv'), index = False)

    test_prediction_calibrated = ir.transform( test_prediction )
    test_preds_df = pd.DataFrame()
    test_preds_df['true y'] = test_y
    test_preds_df['score y'] = test_prediction
    test_preds_df['score y calibrated'] = test_prediction_calibrated
    #test_preds_df.to_csv(os.path.join(output_dir, 'test_preds.csv'), index = False)

    opt_threshold, df_varying_threshold = Select_Threshold_calibrated(valid_preds_df)
    print("Optimal threshold: ", opt_threshold)
    df_varying_threshold.to_csv(os.path.join(out_dir, 'varying_threshold.csv'), index = False)
    


    output_pred_file=f"{name}test_predictions{epoch}.csv"
    output_metrics_file=f"{name}test_metrics{epoch}.csv"

    y_pred_prob = test_prediction
    y_pred_prob_calibrated = test_prediction_calibrated

    metrics = print_metrics_binary(test_y, y_pred_prob_calibrated, threshold=opt_threshold)
    metrics_df = pd.DataFrame([metrics])
    y_pred = (y_pred_prob_calibrated >= opt_threshold).astype(int)
    results_df = pd.DataFrame({
        'Name': test_names.tolist(),              # Names from the provided dataset
        'True_Y': test_y.tolist(),           # True labels
        'Predicted_Probabilities': y_pred_prob.ravel().tolist(),  # Predicted probabilities
        'Predicted_Probabilities (calibrated)': y_pred_prob_calibrated,  # Predicted probabilities
        'Predicted_Y': y_pred.ravel().tolist()       # Predicted class
    })
    results_df.to_csv(os.path.join(out_dir, output_pred_file), index=False)
    print(f"Predictions saved to {output_pred_file}")

    metrics_df.T.to_csv(os.path.join(out_dir, output_metrics_file))
    print(f"Metrics saved to {output_metrics_file}")




    output_pred_file=f"{name}val_predictions{epoch}.csv"
    output_metrics_file=f"{name}val_metrics{epoch}.csv"

    y_pred_prob = val_prediction
    y_pred_prob_calibrated = val_prediction_calibrated

    metrics = print_metrics_binary(val_y, y_pred_prob_calibrated, threshold=opt_threshold)
    metrics_df = pd.DataFrame([metrics])
    y_pred = (y_pred_prob_calibrated >= opt_threshold).astype(int)
    results_df = pd.DataFrame({
        'Name': val_names.tolist(),              # Names from the provided dataset
        'True_Y': val_y.tolist(),           # True labels
        'Predicted_Probabilities': y_pred_prob.ravel().tolist(),  # Predicted probabilities
        'Predicted_Probabilities (calibrated)': y_pred_prob_calibrated,  # Predicted probabilities
        'Predicted_Y': y_pred.ravel().tolist()       # Predicted class
    })
    results_df.to_csv(os.path.join(out_dir, output_pred_file), index=False)
    print(f"Predictions saved to {output_pred_file}")

    metrics_df.T.to_csv(os.path.join(out_dir, output_metrics_file))
    print(f"Metrics saved to {output_metrics_file}")




    output_pred_file=f"{name}train_predictions{epoch}.csv"
    output_metrics_file=f"{name}train_metrics{epoch}.csv"

    y_pred_prob = train_prediction
    y_pred_prob_calibrated = train_prediction_calibrated

    metrics = print_metrics_binary(train_y, y_pred_prob_calibrated, threshold=opt_threshold)
    metrics_df = pd.DataFrame([metrics])
    y_pred = (y_pred_prob_calibrated >= opt_threshold).astype(int)
    results_df = pd.DataFrame({
        'Name': train_names.tolist(),              # Names from the provided dataset
        'True_Y': train_y.tolist(),           # True labels
        'Predicted_Probabilities': y_pred_prob.ravel().tolist(),  # Predicted probabilities
        'Predicted_Probabilities (calibrated)': y_pred_prob_calibrated,  # Predicted probabilities
        'Predicted_Y': y_pred.ravel().tolist()       # Predicted class
    })
    results_df.to_csv(os.path.join(out_dir, output_pred_file), index=False)
    print(f"Predictions saved to {os.path.join(out_dir, output_pred_file)}")

    metrics_df.T.to_csv(os.path.join(out_dir, output_metrics_file))
    print(f"Metrics saved to {os.path.join(out_dir, output_metrics_file)}")






# MIMIC III
# train_X = np.load("Data/Preprocessed_data/Train_X.npy")  
# train_y = np.load("Data/Preprocessed_data/Train_Y.npy") 
# train_names = np.load("Data/Preprocessed_data/Train_names.npy")  

# val_X = np.load("Data/Preprocessed_data/Val_X.npy")  
# val_y = np.load("Data/Preprocessed_data/Val_Y.npy") 
# val_names = np.load("Data/Preprocessed_data/Val_names.npy")  

# test_X = np.load("Data/Preprocessed_data/Test_X.npy")  
# test_y = np.load("Data/Preprocessed_data/Test_Y.npy") 
# test_names = np.load("Data/Preprocessed_data/Test_names.npy")  

# MIMIC - race
# train_X = np.load("Data/subgroup/ihm_race/Preprocessed_data/train_X.npy")  
# train_y = np.load("Data/subgroup/ihm_race/Preprocessed_data/train_Y.npy")  
# train_names = np.load("Data/subgroup/ihm_race/Preprocessed_data/train_names.npy")  

# val_X = np.load("Data/subgroup/ihm_race/Preprocessed_data/val_X.npy")  
# val_y = np.load("Data/subgroup/ihm_race/Preprocessed_data/val_Y.npy")
# val_names = np.load("Data/subgroup/ihm_race/Preprocessed_data/val_names.npy") 

# test_X = np.load("Data/subgroup/ihm_race/Preprocessed_data/test_X.npy")  
# test_y = np.load("Data/subgroup/ihm_race/Preprocessed_data/test_Y.npy")
# test_names = np.load("Data/subgroup/ihm_race/Preprocessed_data/test_names.npy") 

## MIMIC -Gender
# train_X = np.load("Data/subgroup/ihm_gender/Preprocessed_data/train_X.npy")  
# train_y = np.load("Data/subgroup/ihm_gender/Preprocessed_data/train_Y.npy")  
# train_names = np.load("Data/subgroup/ihm_gender/Preprocessed_data/train_names.npy")  

# val_X = np.load("Data/subgroup/ihm_gender/Preprocessed_data/val_X.npy")  
# val_y = np.load("Data/subgroup/ihm_gender/Preprocessed_data/val_Y.npy")
# val_names = np.load("Data/subgroup/ihm_gender/Preprocessed_data/val_names.npy") 

# test_X = np.load("Data/subgroup/ihm_gender/Preprocessed_data/test_X.npy")  
# test_y = np.load("Data/subgroup/ihm_gender/Preprocessed_data/test_Y.npy")
# test_names = np.load("Data/subgroup/ihm_gender/Preprocessed_data/test_names.npy") 

## MIMIC - race and gender
# train_X = np.load("Data/subgroup/ihm_gender_race/Preprocessed_data/train_X.npy")  # Random input data
# train_y = np.load("Data/subgroup/ihm_gender_race/Preprocessed_data/train_Y.npy")  # Binary target
# train_names = np.load("Data/subgroup/ihm_gender_race/Preprocessed_data/train_names.npy")  # Binary target

# val_X = np.load("Data/subgroup/ihm_gender_race/Preprocessed_data/val_X.npy")  # Random validation data
# val_y = np.load("Data/subgroup/ihm_gender_race/Preprocessed_data/val_Y.npy")  # Binary validation target
# val_names = np.load("Data/subgroup/ihm_gender_race/Preprocessed_data/val_names.npy")  # Binary validation target

# test_X = np.load("Data/subgroup/ihm_gender_race/Preprocessed_data/test_X.npy")  
# test_y = np.load("Data/subgroup/ihm_gender_race/Preprocessed_data/test_Y.npy")
# test_names = np.load("Data/subgroup/ihm_gender_race/Preprocessed_data/test_names.npy") 

## eICU
# train_X = np.load("Data/eICU/Preprocessed_data/train_X.npy")  
# train_y = np.load("Data/eICU/Preprocessed_data/train_Y.npy") 
# train_names = np.load("Data/eICU/Preprocessed_data/train_names.npy")  

# val_X = np.load("Data/eICU/Preprocessed_data/val_X.npy")  
# val_y = np.load("Data/eICU/Preprocessed_data/val_Y.npy") 
# val_names = np.load("Data/eICU/Preprocessed_data/val_names.npy")  

# test_X = np.load("Data/eICU/Preprocessed_data/test_X.npy")  
# test_y = np.load("Data/eICU/Preprocessed_data/test_Y.npy") 
# test_names = np.load("Data/eICU/Preprocessed_data/test_names.npy")   


## eICU - rance and gender
# train_X = np.load("Data/subgroup/eicu/ihm_gender_race/Preprocessed_data/train_X.npy")  # Random input data
# train_y = np.load("Data/subgroup/eicu/ihm_gender_race/Preprocessed_data/train_Y.npy")  # Binary target
# train_names = np.load("Data/subgroup/eicu/ihm_gender_race/Preprocessed_data/train_names.npy")  # Binary target

# val_X = np.load("Data/subgroup/eicu/ihm_gender_race/Preprocessed_data/val_X.npy")  # Random validation data
# val_y = np.load("Data/subgroup/eicu/ihm_gender_race/Preprocessed_data/val_Y.npy")  # Binary validation target
# val_names = np.load("Data/subgroup/eicu/ihm_gender_race/Preprocessed_data/val_names.npy")  # Binary validation target

# test_X = np.load("Data/subgroup/eicu/ihm_gender_race/Preprocessed_data/test_X.npy")  
# test_y = np.load("Data/subgroup/eicu/ihm_gender_race/Preprocessed_data/test_Y.npy")
# test_names = np.load("Data/subgroup/eicu/ihm_gender_race/Preprocessed_data/test_names.npy") 

## eICU - gender
# train_X = np.load("Data/subgroup/eicu/ihm_gender/Preprocessed_data/train_X.npy")  
# train_y = np.load("Data/subgroup/eicu/ihm_gender/Preprocessed_data/train_Y.npy")  
# train_names = np.load("Data/subgroup/eicu/ihm_gender/Preprocessed_data/train_names.npy")  

# val_X = np.load("Data/subgroup/eicu/ihm_gender/Preprocessed_data/val_X.npy")  
# val_y = np.load("Data/subgroup/eicu/ihm_gender/Preprocessed_data/val_Y.npy")
# val_names = np.load("Data/subgroup/eicu/ihm_gender/Preprocessed_data/val_names.npy") 

# test_X = np.load("Data/subgroup/eicu/ihm_gender/Preprocessed_data/test_X.npy")  
# test_y = np.load("Data/subgroup/eicu/ihm_gender/Preprocessed_data/test_Y.npy")
# test_names = np.load("Data/subgroup/eicu/ihm_gender/Preprocessed_data/test_names.npy")

# eICU - race
train_X = np.load("Data/subgroup/eicu/ihm_race/Preprocessed_data/train_X.npy")  
train_y = np.load("Data/subgroup/eicu/ihm_race/Preprocessed_data/train_Y.npy")  
train_names = np.load("Data/subgroup/eicu/ihm_race/Preprocessed_data/train_names.npy")  

val_X = np.load("Data/subgroup/eicu/ihm_race/Preprocessed_data/val_X.npy")  
val_y = np.load("Data/subgroup/eicu/ihm_race/Preprocessed_data/val_Y.npy")
val_names = np.load("Data/subgroup/eicu/ihm_race/Preprocessed_data/val_names.npy") 

test_X = np.load("Data/subgroup/eicu/ihm_race/Preprocessed_data/test_X.npy")  
test_y = np.load("Data/subgroup/eicu/ihm_race/Preprocessed_data/test_Y.npy")
test_names = np.load("Data/subgroup/eicu/ihm_race/Preprocessed_data/test_names.npy") 



# Build the model architecture
# load model architecture and weights

#model_path = "Models/Transformer_2/Trial_1/checkpoints/model_epoch_06.keras" 
#model_path = "Models/Transformer_2/Trial_2/checkpoints/model_epoch_03.keras" 
#model_path = "Models/Transformer_2/Trial_3/checkpoints/model_epoch_10.keras" 

# model_path = "Models/Transformer_subgroup/ihm_race/Trial_3/checkpoints/model_epoch_05.keras"
# model_path = "Models/Transformer_subgroup/ihm_race/Trial_3/checkpoints/model_epoch_08.keras"
# model_path = "Models/Transformer_subgroup/ihm_race/Trial_3/checkpoints/model_epoch_06.keras"

# model_path =  "Models/Transformer_subgroup/ihm_gender/Trial_3/checkpoints/model_epoch_09.keras"
# model_path =  "Models/Transformer_subgroup/ihm_gender/Trial_3/checkpoints/model_epoch_06.keras"
# model_path =  "Models/Transformer_subgroup/ihm_gender/Trial_3/checkpoints/model_epoch_06.keras"

# model_path =  "Models/Transformer_subgroup/ihm_race_gender/Trial_1/checkpoints/model_epoch_03.keras"
# model_path =  "Models/Transformer_subgroup/ihm_race_gender/Trial_2/checkpoints/model_epoch_07.keras"
# model_path =  "Models/Transformer_subgroup/ihm_race_gender/Trial_3/checkpoints/model_epoch_03.keras"

# model_path =  "Models/Transformer_subgroup/eicu_ihm_race_gender/Trial_1/checkpoints/model_epoch_16.keras"
# model_path =  "Models/Transformer_subgroup/eicu_ihm_race_gender/Trial_2/checkpoints/model_epoch_14.keras"
# model_path =  "Models/Transformer_subgroup/eicu_ihm_race_gender/Trial_3/checkpoints/model_epoch_10.keras"

# model_path = "Models/Transformer_subgroup/eicu_ihm_gender/Trial_1/checkpoints/model_epoch_18.keras"
# model_path = "Models/Transformer_subgroup/eicu_ihm_gender/Trial_2/checkpoints/model_epoch_19.keras"
# model_path = "Models/Transformer_subgroup/eicu_ihm_gender/Trial_3/checkpoints/model_epoch_07.keras"

# model_path = "Models/Transformer_subgroup/eicu_ihm_race/Trial_3/checkpoints/model_epoch_13.keras"
# model_path = "Models/Transformer_subgroup/eicu_ihm_race/Trial_3/checkpoints/model_epoch_17.keras"
model_path = "Models/Transformer_subgroup/eicu_ihm_race/Trial_3/checkpoints/model_epoch_07.keras"


model = load_model(model_path)
model.summary()

out_dir =    "Models/Transformer_subgroup/eicu_ihm_race/Trial_3"
## use this test for oringinal test set
test_model(model, 
            train_X, train_y, train_names,
            val_X, val_y, val_names,
            test_X, test_y, test_names,
            out_dir=out_dir)


