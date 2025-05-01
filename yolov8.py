from ultralytics import YOLO
import cv2
import os

# フォルダ番号を初期化 (グローバル変数)
folder_count = 1

def process_video(video_path):  #def process_video(self,video_path):
    global folder_count  # グローバル変数を使用

    # モデルのロード
    model = YOLO('yolov8x.pt')
    # 出力動画ファイルのパス
    output_path = 'output1/output1.mp4'
    # 野球ボールのクラスID
    baseball_class_id = 32
    # 検出時のFPS
    detection_fps = 30
    # 通常時のFPS
    normal_fps = 1
    # 検出フラグ
    detected = False
    # ボールが検出されなかった連続回数
    no_detection_count = 0

    # 動画の読み込み
    cap = cv2.VideoCapture(video_path)

    # 動画の幅と高さ取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, detection_fps, (width, height))

    # フレームカウント
    frame_count = 0

    # 出力画像フォルダの作成
    if not os.path.exists('output1'):
        os.makedirs('output1')

    # フォルダが存在しない場合作成
    while True:
        crop_folder = f'crop/crop{folder_count}'
        if not os.path.exists(crop_folder):
            #追加コード
            #self.crop_folder = crop_folder  # crop_folder 属性に値を格納
            os.makedirs(crop_folder)
            break  # フォルダ作成が成功したらループを抜ける
        folder_count += 1  # 既存のフォルダ名と重複したら、folder_count をインクリメント
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 野球ボールが検出されている場合のみ、FPSを高くする
        if detected:
            current_fps = detection_fps
        else:
            current_fps = normal_fps

        # FPS制御
        if frame_count % (int(cap.get(cv2.CAP_PROP_FPS)) // current_fps) != 0:
            continue

        # YOLOv8で推論
        results = model.track(frame, classes=[baseball_class_id], conf=0.3, max_det=1) 

        # 検出結果の処理
        detected_in_frame = False
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = box.cls[0].item()
                if cls == baseball_class_id:
                    detected_in_frame = True
                    detected = True
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # ボールをくり抜く
                    cropped_image = frame[y1:y2, x1:x2]

                    # ファイル名が重複しないように、連番を付加するなど
                    count = 1
                    while os.path.exists(f'{crop_folder}/crop{frame_count}_{count}.jpg'):
                        count += 1

                    file_path = f'{crop_folder}/crop{frame_count}_{count}.jpg'
                    cv2.imwrite(file_path, cropped_image)

        # ボールが検出されなかった場合の処理
        if not detected_in_frame:
            no_detection_count += 1
            if no_detection_count >= 5:
                detected = False
                no_detection_count = 0
        else:
            no_detection_count = 0

        # 出力動画にフレームを書き込む
        out.write(frame)

    # 後処理
    cap.release()
    out.release()

    print('処理が完了しました。')