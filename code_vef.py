from PIL import Image, ImageDraw, ImageFont, ImageFilter
import  random
import cv2
import  numpy as np
import  matplotlib.pyplot as plt
path =  'C://ProgramData//Anaconda3//Lib//site-packages//matplotlib//mpl-data//fonts//ttf//'
data_path = 'D://pytorch//pokemen//code_vef//'

def rndChar():
    return chr(random.randint(65, 90))

def rndInt():
    return str(random.randint(0,9))

def rndColor():
    return (random.randint(64, 255), random.randint(64, 255),random.randint(64, 255))

def rndColor2():
    return (random.randint(32, 127), random.randint(32, 127), random.randint(32, 127))

def gaussian_noise():
    mu = 225
    sigma = 20
    return  tuple((np.random.normal(mu, sigma,3).astype(int)))

def rotate(x, angle):
    M_rorate = cv2.getRotationMatrix2D((x.shape[0]/2, x.shape[1]/2),angle,1)
    x = cv2.warpAffine(x, M_rorate, (x.shape[0], x.shape[1]))
    return  x

width = 180*4
height = 180


def gen_image(num):
    for l in range(num):

        image = Image.new('RGB', (width, height), (255, 255, 255))  # 先生成一张大图
        #Image._show(image)
        font = ImageFont.truetype(path + 'cmb10.ttf', 36)

        draw = ImageDraw.Draw(image)  # 新的画板
        #Image._show(image)
        for x in range(0, width):
            for y in range(0, height):
                draw.point((x, y), fill=rndColor())
        #Image._show(image)
        label = []

        for t in range(4):  # 每一张验证码4个数字
            numb = rndInt()
            draw.text((180 * t + 60 + 10, 60 + 10), numb, font=font, fill=rndColor2())
            label.append(numb)
            #print(label)
        #Image._show(image)
        with open(data_path + "label_val.txt", "a") as f:
            for s in label:
                #print(s)
                f.write(s + ' ')
            f.writelines("\n")  # 写入label

        img = image.filter(ImageFilter.GaussianBlur(radius=0.5))
        img = np.array(img)
        #print(img.shape)
        #plt.imshow(img)
        #c = img[:,0:180]

        img1 = np.array([])

        for i in range(0, 4):
            img0 = img[:, 180 * i: 180 * i + 180]  # 提取含有验证码的小图
            # cv2.imshow("dows", img0)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            #img[:,0:180]  img[:,180:360] img[:,360,540]  img[:,540:720]
            #plt.imshow(img)
            #plt.imshow(img0)
            # plt.show(img)
            # plt.show(img0)

            angle = random.randint(-45, 45)
            img0 = rotate(img0, angle)  # 对小图随机旋转

            if img1.any():
                img1 = np.concatenate((img1, img0[60:120, 60:120, :]), axis=1)

            else:
                img1 = img0[60:120, 60:120, :]

        plt.imsave(data_path + 'src_val/' + str(l) + '.jpg', img1)  # 保存结果
if __name__ == '__main__':
        gen_image(96)
