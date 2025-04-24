# https://digi.bib.uni-mannheim.de/tesseract/
# 配置环境变量如E:\Program Files (x86)\Tesseract-OCR
# tesseract -v进行测试
# tesseract XXX.png 得到结果 
# pip install pytesseract
# anaconda lib site-packges pytesseract pytesseract.py
# tesseract_cmd 修改为绝对路径即可
from PIL import Image
import pytesseract
import cv2
import os

preprocess = 'blur'  # thresh

image = cv2.imread('../img/scan.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

'''
预处理方法	适用场景	优点	缺点
Otsu阈值化	光照不均的文档/高对比度物体	自动适应阈值，无需手动调参	对复杂背景效果较差
中值滤波	图像中有散粒噪声（如老旧照片）	有效去噪且保护边缘	可能模糊细小纹理

'''
# 自动阈值化
if preprocess == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# 中值滤波法
if preprocess == "blur":
    gray = cv2.medianBlur(gray, 3)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

text = pytesseract.image_to_string(Image.open(filename))
print(text)
os.remove(filename)

cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
