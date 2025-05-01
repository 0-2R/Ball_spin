import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Add
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import TensorBoard,ModelCheckpoint,EarlyStopping

#ハイパーパラメーター
hp1 = {}
hp1['class_num'] = 4 # クラス数（今回は0°,45°,90°,135°の4クラス）
hp1['batch_size'] = 32 # バッチサイズ
hp1['epoch'] = 20 #エポック数

#データセットのロード
##前章で作ったデータセットをここで読み込む
X_train = np.load("./X_train.npy")
X_test = np.load("./X_test.npy")
y_train = np.load("./y_train.npy")
y_test = np.load("./y_test.npy")

#入力サイズ
input_shape=X_train.shape[1:]

# CNNを構築
def CNN(input_shape):
        model = Sequential()
        
        model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(32, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
                
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(128, (3, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(hp1['class_num']))
        model.add(Activation('softmax'))

        return model

#モデルを選択
model=CNN(input_shape)

#コンパイル
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
 
#データの記録
log_dir = os.path.join(os.path.dirname(__file__), "logdir")
model_file_name="ball_predict.keras"

# EarlyStoppingの設定
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

#訓練
history = model.fit(
        X_train, y_train,
         epochs=hp1['epoch'],
         verbose=1,
         validation_split = 0.2,
         callbacks=[
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(os.path.join(log_dir,model_file_name),save_best_only=True),
                early_stopping
                ],
)

# 損失のグラフ
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 正解率のグラフ
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#評価 & 評価結果出力
loss, accuracy = model.evaluate(X_test, y_test, batch_size=hp1['batch_size'])