from tensorflow import keras, config
import cv2
import numpy as np

gpus = config.experimental.list_physical_devices('GPU')
if gpus:
    [config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

model = keras.models.load_model("poker_model_1.h5")

# 丟進你想要的照片或圖
img = cv2.imread("test/S2/space2-2.jpg", cv2.IMREAD_GRAYSCALE)
# 讓值從0~255到0~1之間
img = img.reshape(1, 128, 128, 1) / 255
# 抓52牌中最大機率的牌
ans = np.argmax(model.predict(img)[0])
print("ans=%d" % ans)
# 代表數字幾
num = (ans % 13)
if num == 0:
    num = 1
elif num <= 4:
    num += 9
else:
    num -= 3

# 代表第幾個花色
if ans in range(13):
    print("梅花%d" % num)
elif ans in range(13, 26):
    print("方塊%d" % num)
elif ans in range(26, 39):
    print("紅心%d" % num)
elif ans in range(39, 52):
    print("黑桃%d" % num)
