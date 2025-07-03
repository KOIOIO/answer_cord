import cv2
import imutils
from imutils.perspective import four_point_transform
import numpy as np

# 读入图片
# image = cv2.imread("img2.jpg")

def getFourPtTrans(img):
    '''
    返回最大矩形框的极点坐标矩阵(4,2)

    原理：
    1. 首先将彩色图像转换为灰度图像，减少颜色通道的干扰，简化后续处理。
    2. 对灰度图像进行高斯滤波，平滑图像，去除噪声，避免噪声对边缘检测的影响。
    3. 使用自适应二值化方法将图像转换为二值图像，使得目标区域和背景区域有明显的区分。
    4. 利用 Canny 边缘检测算法检测图像中的边缘。
    5. 查找图像中的轮廓，并按面积大小降序排序。
    6. 对排序后的轮廓进行近似处理，若近似轮廓有四个顶点，则认为找到了最大矩形框。
    '''
    # 转换为灰度图像，减少颜色信息，降低计算复杂度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波，通过高斯核函数对图像进行卷积操作，平滑图像，去除噪声
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    # 自适应二值化方法，根据图像局部区域的像素分布计算阈值，将图像转换为二值图像
    blurred=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)

    # canny边缘检测，通过多阶段算法检测图像中的边缘，包括高斯滤波、梯度计算、非极大值抑制和双阈值检测
    #edged = cv2.Canny(blurred, 10, 100)
    #edged = cv2.Canny(blurred, 110, 100)
    edged = cv2.Canny(blurred, 10, 10)
    #edged = cv2.Canny(blurred, 1, 100)


    # 从边缘图中寻找轮廓，cv2.RETR_EXTERNAL 表示只检测外部轮廓，cv2.CHAIN_APPROX_SIMPLE 表示压缩水平、垂直和对角线方向的冗余点
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # 根据 OpenCV 版本不同，获取正确的轮廓列表
    cnts = cnts[1] if imutils.is_cv3() else cnts[0]
    docCnt = None

    # 确保至少有一个轮廓被找到
    if len(cnts) > 0:
        # 将轮廓按大小降序排序，优先处理面积大的轮廓
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # 对排序后的轮廓循环处理
        for c in cnts:
            # 获取近似的轮廓，cv2.approxPolyDP 使用 Douglas-Peucker 算法对轮廓进行多边形逼近
            #维基百科：ε的选择通常由用户定义。像大多数线拟合 / 多边形逼近 / 主点检测方法一样，它可以通过使用数字化 / 量化引起的误差边界作为终止条件来实现非参数化。[1]
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
            if len(approx) == 4:
                docCnt = approx
                break


    docCnt=docCnt.reshape(4,2)
    return docCnt

def getXY(docCnt):
    '''return (minX,minY,maxX,maxY)

    原理：遍历矩形框的四个顶点，找出最小的 x、y 坐标和最大的 x、y 坐标，从而确定矩形框的边界。
    '''
    minX,minY=docCnt[0]
    maxX,maxY=docCnt[0]
    for i in range(1,4):
        minX=min(minX,docCnt[i][0])
        maxX=max(maxX,docCnt[i][0])
        minY=min(minY,docCnt[i][1])
        maxY=max(maxY,docCnt[i][1])
    return minX,minY,maxX,maxY

# 判断题号
def judgeQ(x,y):
    '''传入时记得x+1,y+1

    原理：根据传入的 x、y 坐标，按照预设的规则计算对应的题号。
    当 x 小于 6 时，使用一种计算方式；当 x 大于等于 6 时，使用另一种计算方式。
    '''
    if x<6:
        return x+(y-1)//4*5
    else:
        return ((x-1)//5-1)*25+10+(x-1)%5+1+(y-1)//4*5

# 判断答案
def judgeAns(y):
    '''
    原理：根据传入的 y 坐标，通过取模运算判断对应的答案选项。
    若 y 对 4 取模的结果为 1，则答案为 A；为 2 则为 B；为 3 则为 C；为 0 则为 D。
    '''
    if y%4==1:
        return 'A'
    if y%4==2:
        return 'B'
    if y%4==3:
        return 'C'
    if y%4==0:
        return 'D'

def judge0(x,y):
    '''
    原理：调用 judgeQ 和 judgeAns 函数，分别计算题号和答案选项，并以元组形式返回。
    '''
    return (judgeQ(x,y),judgeAns(y))


# 人工标记答题卡格子的边界坐标
xt1=[0,110,210,310,410,565,713,813,913,1013,1163,1305,1405,1505,1605,1750,1895,1995,2095,2195,2295]
yt1=[0,125,175,225,300,422,471,520,600,716,766,817,902,1012,1064,1113,1195,1309,1357,1409,1479]

def markOnImg(img,width,height):
    '''在四点标记的图片上，将涂黑的选项标记，并返回(图片,坐标)

    原理：
    1. 调用 getFourPtTrans 函数获取最大矩形框的极点坐标。
    2. 将图像转换为灰度图像，并使用四点透视变换将图像校正为正视角度。
    3. 对校正后的灰度图像进行自适应二值化处理，然后调整图像大小。
    4. 对二值图像进行均值滤波，再次二值化处理，以突出显示涂黑的选项。
    5. 查找二值图像中的轮廓，筛选出符合条件的轮廓，计算其质心坐标，并在原图像上标记。
    '''
    docCnt=getFourPtTrans(img)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 四点透视变换，将图像校正为正视角度，方便后续处理
    paper=four_point_transform(img,docCnt)
    warped=four_point_transform(gray,docCnt)

    # 灰度图二值化，使用自适应二值化方法将图像转换为二值图像
    thresh=cv2.adaptiveThreshold(warped,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2)
    # resize，调整图像大小，方便后续处理
    thresh = cv2.resize(thresh, (width, height), cv2.INTER_LANCZOS4)
    paper = cv2.resize(paper, (width, height), cv2.INTER_LANCZOS4)
    warped = cv2.resize(warped, (width, height), cv2.INTER_LANCZOS4)

#######
    # 均值滤波，平滑图像，去除噪声
    #  参数为5 出现噪声
    #ChQImg = cv2.blur(thresh, (5,  5))
    #参数为10-15正常
    ChQImg = cv2.blur(thresh, (13, 13))
    #ChQImg = cv2.blur(thresh, (10, 10))
    # 参数大于20检查不出边缘
    #ChQImg = cv2.blur(thresh, (40, 40))
    # 二值化，120是阈值，将图像再次二值化，突出显示涂黑的选项


    ChQImg = cv2.threshold(ChQImg, 120, 225, cv2.THRESH_BINARY)[1]

    Answer=[]

    # 二值图中找答案轮廓，cv2.RETR_TREE 表示检测所有轮廓并重建轮廓树，cv2.CHAIN_APPROX_SIMPLE 表示压缩冗余点
    cnts=cv2.findContours(ChQImg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnts=cnts[1] if imutils.is_cv3() else cnts[0]
    for c in cnts:
        # 获取轮廓的边界矩形
        x,y,w,h=cv2.boundingRect(c)
        # 筛选符合条件的轮廓，即宽度和高度在一定范围内
        if w>50 and h>20 and w<100 and h<100:
            # 计算轮廓的矩，用于计算质心坐标
            M=cv2.moments(c)
            cX=int(M["m10"]/M["m00"])
            cY=int(M["m01"]/M["m00"])

            # 在原图像上绘制轮廓和质心
            cv2.drawContours(paper,c,-1,(0,0,255),5)
            cv2.circle(paper,(cX,cY),7,(255,255,255),2)
            Answer.append((cX,cY))
    return paper,Answer


def solve(imgPath):
    '''
    原理：
    1. 读取指定路径的图像，调用 getFourPtTrans 函数获取答案区域的极点坐标。
    2. 根据极点坐标切取学号区域和科目区域的图像，并分别获取其极点坐标。
    3. 调用 markOnImg 函数处理答案区域、学号区域和科目区域的图像，标记涂黑的选项并获取坐标。
    4. 根据坐标计算题号和答案选项、学号和科目信息。
    5. 调整处理后的图像大小，并返回处理后的图像和相关信息。
    '''
    image=cv2.imread(imgPath)
    # 搞到答案的四点坐标
    ansCnt=getFourPtTrans(image)
    xy=getXY(ansCnt)

    xy=getXY(ansCnt)
    # 切取上半部分的图
    stuNum=image[0:xy[1],xy[0]:xy[2]]
    numCnt=getFourPtTrans(stuNum)


    xy=getXY(numCnt)
    # 切右半部分的图，方便识别科目
    course=image[0:int(xy[3]*1.1),xy[2]:len(image)]


    '''处理答案'''
    width1,height1=2300,1500
    ansImg,Answer=markOnImg(image,width1,height1)

    # [(题号，答题卡上的答案),]
    IDAnswer=[]
    for a in Answer:
        for x in range(0,len(xt1)-1):
            if a[0]>xt1[x] and a[0]<xt1[x+1]:
                for y in range(0,len(yt1)-1):
                    if a[1]>yt1[y] and a[1]<yt1[y+1]:
                        IDAnswer.append(judge0(x+1,y+1))

    IDAnswer.sort()
    ansImg=cv2.resize(ansImg,(600,400))
    # cv2.imshow("answer",ansImg)
    # cv2.imwrite("answer.jpg",ansImg)
    # print(IDAnswer)


    '''处理学号'''
    width2,height2=1000,1000
    numImg,Answer=markOnImg(stuNum,width2,height2)

    Answer.sort()
    # xt2=list(range(0,1100,100))
    yt2=[227,311,374,442,509,577,644,711,781,844]

    NO=''
    for a in Answer:
        for y in range(len(yt2)-1):
            if a[1]>yt2[y] and a[1]<yt2[y+1]:
                NO+=str(y)
    if NO=='':
        NO="Nan"
    numImg=cv2.resize(numImg,(300,200))
    # cv2.imshow('SNO',numImg)
    # print(NO)

    '''处理科目'''
    width3,height3=300,1000
    courseImg,Answer=markOnImg(course,width3,height3)
    yt3=list(range(250,1000,65))
    course_list=['政治','语文','数学','物理','化学','英语',
        '历史','地理','生物','文综','理综']

    s=-1
    if len(Answer)>0:
        for y in range(len(yt3)-1):
            if Answer[0][1]>yt3[y] and Answer[0][1]<yt3[y+1]:
                s=y

    courseImg=cv2.resize(courseImg,(150,400))
    # cv2.imshow('course',courseImg)
    course_checked="Nan"
    if s!=-1:
        course_checked=course_list[s]


    # cv2.waitKey()
    # cv2.destroyAllWindows()
    return ((numImg,courseImg,ansImg),(NO,course_checked,IDAnswer))

if __name__ == "__main__":
    (numImg,courseImg,ansImg),(NO,course,IDAnswer)=solve("img2.jpg")
    cv2.imshow("answer",ansImg)
    cv2.imshow('SNO',numImg)
    cv2.imshow('course',courseImg)
    print(NO)
    print(course)
    print(IDAnswer)
    cv2.waitKey()
    cv2.destroyAllWindows()
