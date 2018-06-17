import os
import os.path
import cv2
import glob
import imutils

CAPTCHA_IMAGE_FOLDER = 'train_images'
OUTPUT_FOLDER = 'extracted_images'

# 获取需要处理的验证码的路径
train_images = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, '*'))
counts = {}

# 遍历图片
# enumerate可以同时返回列表下标和值
for (i, train_image) in enumerate(train_images):
    print('processing image {}/{}'.format(i+1, len(train_images)))
    # 获取图片文件名，取得图片文字内容
    filename = os.path.basename(train_image)
    captcha_content = os.path.splitext(filename)[0]
    img = cv2.imread(train_image)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 给源图像增加边界，复制边缘
    gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
    # 设定二值化的阈值,0~255级灰度
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # 标记每个字母轮廓
    contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 根据cv的版本选择轮廓对象本身还是，轮廓的属性
    contours = contours[0] if imutils.is_cv2() else contours[1]
    # 单个字母
    letter_images = []
    # 循环提取图片中的每个字母
    for contour in contours:
        # 获取包括字母的最小矩形的左上角坐标x，y，以及举行的宽w，高h
        (x, y, w, h) = cv2.boundingRect(contour)
        # 通过对宽高比来判断是否把相连的两个字母判断为一个字母
        # 设定一定的宽高比来进行字母拆分
        if w / h > 1.25:
            half_width = int(w / 2)
            letter_images.append((x, y, half_width, h))
            letter_images.append((x + half_width, y, half_width, h))
        else:
            letter_images.append((x, y, w, h))
        # 判断是否识别出了4个字母
        if len(letter_images) != 4:
            continue
        # 确保识别的字母图按照原本图中顺序排列
        letter_images = sorted(letter_images, key=lambda x: x[0])
        # 保存字母图的每个字母，用zip打包使得同时取得字母图和对应的文件名中的字母
    for letter_image_box, letter_text in zip(letter_images, captcha_content):
        x, y, w, h = letter_image_box
        # 取略大于字母边缘的字母图像,设定为大于字母边缘2像素
        letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
        # 选择字母保存位置
        save_path = os.path.join(OUTPUT_FOLDER, letter_text)
        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # 为每个字母记录出现次数
        count = counts.get(letter_text, 1)
        # 重命名
        p = os.path.join(save_path, '{}.png'.format(str(count).zfill(6)))
        cv2.imwrite(p, letter_image)
        # 为每个字母记录出现次数
        counts[letter_text] = count + 1
