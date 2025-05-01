import glob
import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

classes = ["0","45","90","135"]###判別したいラベルを入力
num_classes = len(classes)
image_size = 128

#datesetのディレクトリ
datadir='./'

#画像の読み込み
X = []
Y = []

for index, classlabel in enumerate(classes):
    photos_dir = datadir+ classlabel
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):

        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)

        for angle in range(-70, 70, 10):
            # 回転
            img_r = image.rotate(angle)
            data = np.asarray(img_r)
            X.append(data)
            Y.append(index)

X = np.array(X)
Y = np.array(Y)

#２割テストデータへ
(X_train, X_test, y_train, y_test) = train_test_split(X, Y,test_size=0.2)

#正規化
X_train = X_train.astype("float") / 255
X_test = X_test.astype("float") / 255

#教師データの型を変換
y_train = to_categorical(y_train,num_classes)
y_test = to_categorical(y_test, num_classes)

np.save("./X_train.npy", X_train)
np.save("./X_test.npy", X_test)
np.save("./y_train.npy", y_train)
np.save("./y_test.npy", y_test)