import numpy as np
import cv2  # OpenCV を使用
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from sklearn.linear_model import LinearRegression
from keras.models import load_model

def select_folder():
    root = tk.Tk()
    root.withdraw()
    images_folder = filedialog.askdirectory(title="画像フォルダを選択してください")
    # フォルダが選択されたら、処理を実行する
    if images_folder:
        run_cnn(images_folder)

def load_and_preprocess_image(file, image_size=128):
    """画像を読み込み、前処理を行う関数"""
    image = cv2.imread(file)  # OpenCV で画像を読み込み
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 色空間を変換
    image = cv2.resize(image, (image_size, image_size))  # リサイズ
    image = image.astype('float32') / 255.0  # 正規化
    return image

def predict_angle(model, image, classes):
  """画像の角度を予測する関数"""
  result = model.predict(np.expand_dims(image, axis=0))  # 予測
  predicted_class = np.argmax(result)  # 最も確率の高いクラス
  percentage = int(result[0][predicted_class] * 100)  # 確率
  
  return classes[predicted_class], percentage

def calculate_rotation_per_frame(angle1, angle2):
    """2枚の画像の予測角度から、1フレームあたりの回転数を計算する関数"""
    angle_diff = angle1 - angle2  # 角度差を計算

    if angle_diff<=-135:
         angle_diff+=180

    rotation_per_frame = angle_diff / 360  # 1フレームでangle_diff回転
    return rotation_per_frame

def run_cnn(images_folder):
    """CNNモデルで画像分類を実行する関数"""
    model_path = "./logdir/ball_predict.keras"
    classes = ["0", "45", "90", "135"]
    angle_history = []  # 過去の角度を保存するリスト
    total_rotation = 0  # 累計回転数
    fps = 240  # フレームレート
    frame = 10 #予測に使用するフレーム数
    frame_count = -1  # フレーム数をカウントする変数を追加
    plus=0 #最初の角度を135にするための変数
    count0 = 0 #ajusted_angleが0のカウンター

    # モデルの読み込み
    model = load_model(model_path)

    # 画像ファイルのパスを取得
    #最終更新日時,reverseは降順でソート
    image_files = sorted(glob.glob(images_folder + "/*.jpg"),key=os.path.getmtime,reverse=True)

    rotation_per_frame_history = []  # rotation_per_frameを保存するリスト

    for i, file in enumerate(image_files):
        frame_count += 1  # フレーム数をインクリメント
        # 画像の読み込みと前処理
        image = load_and_preprocess_image(file)

        # 角度の予測
        predicted_angle, percentage = predict_angle(model, image, classes)

        # 最初の画像の場合のみ、plusを計算
        if i == 0:
            target_angle = 135
            plus = target_angle - int(predicted_angle)

        # 角度にplusを加算
        adjusted_angle = (int(predicted_angle) + plus) % 180
        angle_history.append(adjusted_angle)  # 調整後の角度をリストに追加

        # 角度の履歴が2枚以上になったら、累積回転数を計算
        if len(angle_history) >= 2:
            rotation_per_frame = calculate_rotation_per_frame(angle_history[-2], angle_history[-1])  # このフレームでの回転数を計算
            total_rotation += rotation_per_frame  # 累積回転数を更新
            rotation_per_frame_history.append(rotation_per_frame)  # rotation_per_frameをリストに追加

        # 結果の出力
        print(f"ファイル名: {Path(file).name}")
        print(f"予測角度: {predicted_angle} ({percentage}%), "\
            f"累積回転数:{total_rotation:.4f}回転, フレーム数:{frame_count}")

        if  count0 < 1: #調整後の予測角度に0が出ていないなら
            if len(angle_history) >=1: # rotation_per_frame_historyに要素が1つ以上ある場合のみ実行
                x = np.arange(len(angle_history)).reshape(-1, 1)  # フレーム番号
                y = np.array(angle_history)  # rotation_per_frame

                reg = LinearRegression().fit(x, y)  # 線形回帰モデルの学習
                rotation_per_frame_reg = reg.coef_[0] / 360 # 1フレームあたりの回転 (回帰係数)

                # rpm2の計算
                rpm2 = -rotation_per_frame_reg * fps * 60

                print(f"1フレームあたりの回転 (線形回帰): {rotation_per_frame_reg:.4f}回転")
                #print(f"回転数（線形回帰）: {rpm2:.4f}rpm")  # rpmを出力

                #ajusted_angleが0のときカウント+1
                if adjusted_angle == 0: 
                    count0+=1
            
         # 累計回転数が指定した回転数になったら、1回転にかかるフレーム数を計算・出力
        if total_rotation >= 0.75:#変更可
        #if (i + 1) == frame:  # frame分の処理が完了したら
            if total_rotation !=0: # total_rotationが0だとZeroDivisionErrorになるため
                rpm = 60*(fps/frame_count)*total_rotation #1フレームにかかるfps

                print(f"回転数（方法1）: {rpm:.4f}rpm")
                print(f"回転数（方法2）（線形回帰）: {rpm2:.4f}rpm")  # 線形回帰のrpmを出力

            else:
                print("回転数が0のため、rpmを計算できません。")
            break #ループを抜ける
    
    # rotation_per_frameのプロット
    plt.plot(x, y, color='red', marker='o',linestyle='none', label="angle")  
    
    # 回帰直線のプロット
    plt.plot(x, reg.predict(x), label="Regression Line")  
    
    # 縦軸の目盛りを設定
    plt.yticks([0, 45, 90, 135])
    plt.xlabel("Frame")
    plt.ylabel("Angle")
    
    #凡例
    plt.legend()
    plt.show()
        
if __name__ == "__main__":
    select_folder()