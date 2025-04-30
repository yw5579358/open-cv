#导入工具包
from collections import OrderedDict
import numpy as np
import argparse
import dlib
import cv2

#https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
#http://dlib.net/files/

# 参数
##需要将numpy降低到1.4一下，2版本和dlib兼容不好！！
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=True,
# 	help="path to facial landmark predictor")
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image")
# args = vars(ap.parse_args())

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])


FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])
# 将 dlib 的 shape 对象（包含了面部关键点的坐标）转换为一个 NumPy 数组
def shape_to_np(shape, dtype="int"):
	# 创建68*2
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# 遍历每一个关键点
	# 得到坐标
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords
'''
该函数负责在图像上可视化面部关键点区域。它首先创建两个图像副本，一个用于覆盖绘制区域（overlay），另一个用于最终显示（output）。然后，它遍历每个面部区域（如嘴巴、眼睛等）并绘制区域的形状：

对于下巴区域，它使用 cv2.line() 连线；

对于其他区域（如眼睛、鼻子等），它使用 cv2.drawContours() 绘制凸包。

最后，使用 cv2.addWeighted() 将绘制的区域叠加回原始图像，达到透明叠加效果
'''
def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# 创建两个copy
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()
	# 设置一些颜色区域
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]
	# 遍历每一个区域
	for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
		# 得到每一个点的坐标
		(j, k) = FACIAL_LANDMARKS_68_IDXS[name]
		pts = shape[j:k]
		# 检查位置
		if name == "jaw":
			# 用线条连起来
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)
		# 计算凸包
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)
	# 叠加在原图上，可以指定比例
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	return output

# 加载人脸检测与关键点定位
predictor = dlib.shape_predictor('D:/wym/work/my/home/open-cv/21-dect-face/shape_predictor_68_face_landmarks.dat')

# 读取输入数据，预处理
image = cv2.imread('D:/wym/work/my/home/open-cv/21-dect-face/images/liudehua.jpg')
if image is None:
    raise ValueError("无法加载图像，请检查路径是否正确。")
(h, w) = image.shape[:2]
width=500
r = width / float(w)
dim = (width, int(h * r))
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 人脸检测
rects = dlib.get_frontal_face_detector()(gray, 1)

# 遍历检测到的框
for (i, rect) in enumerate(rects):
	# 对人脸框进行关键点定位
	# 转换成ndarray
	'''
	过 dlib.get_frontal_face_detector() 进行人脸检测，返回一个包含人脸位置的矩形框（rects）。然后，使用 predictor 对每个检测到的人脸框进行面部关键点定位，并将其转换为 NumPy 数组。
	'''
	shape = predictor(gray, rect)
	shape = shape_to_np(shape)

	# 遍历每一个部分
	#对于每个面部区域（例如嘴巴、眼睛等），首先复制一张图像（clone），然后在该图像上标记区域的名字（例如“嘴巴”）。接着，使用 cv2.circle() 在关键点上绘制圆点。
	for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
		clone = image.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)

		# 根据位置画点
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)

		# 提取ROI区域
		# 对于每个面部区域，使用 cv2.boundingRect() 计算一个矩形框来提取区域（ROI）。然后，调整该区域的大小并显示它
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		
		roi = image[y:y + h, x:x + w]
		(h, w) = roi.shape[:2]
		width=250
		r = width / float(w)
		dim = (width, int(h * r))
		roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)
		
		# 显示每一部分
		cv2.imshow("ROI", roi)
		cv2.imshow("Image", clone)
		cv2.waitKey(0)

	# 展示所有区域
	output = visualize_facial_landmarks(image, shape)
	cv2.imshow("Image", output)
	cv2.waitKey(0)

'''
加载输入图像；

使用 dlib 检测并标记面部关键点；

将关键点分为不同的区域，进行标记和可视化；

提取并显示面部区域（如眼睛、嘴巴）；

最后，将处理结果以可视化图像展示出来。
'''