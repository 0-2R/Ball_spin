import os
import sys

# yolov8.py が格納されているフォルダへの相対パスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ultralytics'))

import yolov8

def main():
    # ファイル選択ダイアログを開く (yolov8.py から移動)
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])

    if video_path:
        # yolov8.py の process_video 関数を呼び出し、切り抜いた画像のパスを取得
        yolov8.process_video(video_path)
            
if __name__ == "__main__":
    main()