import imutils
import cv2

def resize_to_fit(image, width, height):
    (h, w) = image.shape[:2]
    if w > h:
        # 修改宽度
        image = imtuils.resize(image, width=width)
    else:
        # 修改长度
        image = imtuils.resize(image, height=height)
    # 计算需要填充的边缘的大小
    padH = int((height - image.shape[0]) / 2.0)
    padW = int((width - image.shape[1]) / 2.0)
    # 填充图片边缘(参数：图像，上，下，左，右，填充类型)
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW, cv2.BORDER_REPLICATE)
    # 修改宽高
    image = cv2.resize(image, (width, height))
    return image
