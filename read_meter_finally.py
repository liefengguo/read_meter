# -*- coding: utf-8 -*-
"""
Created on Sun Jul  3 19:58:22 2022

@author: ThinkPad
"""


import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy
import torch
import math
from models.net import U2NET
import numpy as np
import numpy as np
import math
import cv2
import argparse
import  pickle
from sympy import *

# 读数后处理中有把圆形表盘转成矩形的操作，矩形的宽即为圆形的外周长
# 因此要求表盘图像大小为固定大小，这里设置为[512, 512]
METER_SHAPE = [512, 512]  # 高x宽
# 圆形表盘的中心点
CIRCLE_CENTER = [256, 256]  # 高x宽
# 圆形表盘的半径
CIRCLE_RADIUS = 250
# 圆周率
PI = 3.1415926536
# 在把圆形表盘转成矩形后矩形的高
# 当前设置值约为半径的一半，原因是：圆形表盘的中心区域除了指针根部就是背景了
# 我们只需要把外围的刻度、指针的尖部保存下来就可以定位出指针指向的刻度
RECTANGLE_HEIGHT = 120
# 矩形表盘的宽，即圆形表盘的外周长
RECTANGLE_WIDTH = 1570
# 当前案例中只使用了两种类型的表盘，第一种表盘的刻度根数为50
# 第二种表盘的刻度根数为32。因此，我们通过预测的刻度根数来判断表盘类型
# 刻度根数超过阈值的即为第一种，否则是第二种
TYPE_THRESHOLD = [70,100]
# 两种表盘的配置信息，包含每根刻度的值，量程，单位
METER_CONFIG = [{
    'scale_interval_value': 110 / 110.0,
    'range': 110.0,
    'unit': "(MPa)"
}, {
    'scale_interval_value': 16 / 80.0,#16 / 80.0
    'range': 16.0,
    'unit': "(MPa)"
},{
    'scale_interval_value': 120 / 60.0,#16 / 80.0
    'range': 120.0,
    'unit': "(MPa)"
}]
# 分割模型预测类别id与类别名的对应关系
SEG_CNAME2CLSID = {'background': 0, 'pointer': 255, 'scale': 255}
METER_distinguish = {'pressure_meter': 16 / (16.1 / 16) / 250 ,'temp_meter_40': 120/(121 /120) /250,'temp_meter_30': 110/(111 /110) /250 } # 比例关系

########################################################################


def line_detect(img, cx = CIRCLE_CENTER[0], cy= CIRCLE_CENTER[1]):
    oimg =img.copy()
    edges = cv2.Canny(oimg, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 6, maxLineGap=4)

    kernel = np.ones((3, 3), np.uint8)


    nmask = np.zeros(img.shape, np.uint8)
    # lines = mential.findline(self=0, cp=[x, y], lines=lines)
    # print('lens', len(lines))
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(nmask, (x1, y1), (x2, y2), 100, 1, cv2.LINE_AA)
    x1, y1, x2, y2 = lines[0][0]
    d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
    d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2
    if d1 > d2:
        axit = [x1, y1]
    else:
        axit = [x2, y2]
    nmask = cv2.erode(nmask, kernel, iterations=1)

    cnts, hier = cv2.findContours(nmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts, hier = cv2.findContours(nmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areass = [cv2.contourArea(x) for x in cnts]
    # print(len(areass))
    i = areass.index(max(areass))


    cnt = cnts[i]
    output = cv2.fitLine(cnt, 2, 0, 0.001, 0.001)
    k = output[1] / output[0]
    k = round(k[0], 2)
    b = output[3] - k * output[2]
    b = round(b[0], 2)
    x1 = cx
    x2 = axit[0]
    y1 = int(k * x1 + b)

    y2 = int(k * x2 + b)
    cv2.line(oimg, (x1, y1), (x2, y2), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(oimg, (x1, y1), (cx,cy), (255, 255, 255), 1, cv2.LINE_AA)



    return x1, y1, x2, y2
def decter_only_line(imgpath):

    img = cv2.imread(imgpath,0)
    nor_img = resize(img,METER_SHAPE)
    cx, cy = CIRCLE_CENTER[0],CIRCLE_CENTER[1]
    da, db, dc, de = line_detect(nor_img, cx, cy)
    print(da,db,dc,de)
    distinguish = 100 / 360
    OZ = [da, db, cx, cy]
    OP = [da, db, dc, de]
    dab = (cx-da)*(cx-da) +(cy-db)*(cy-db)
    dce = (cx-dc)*(cx-dc) +(cy-de)*(cy-de)
    print("差距：",dab ,dce)
    if(dab< dce):
        OZ = [da, db, da, 300]
        OP = [da, db, dc, de]
    cv2.line(nor_img, (da, db), (da, 300), (255,255, 255), 2, cv2.LINE_AA)
    cv2.line(nor_img, (da, db), (dc, de), (255,255, 255), 2, cv2.LINE_AA)
    # cv2.imshow("oimg,",nor_img)
    # cv2.waitKey(0)
    ang1 = angle(OP, OZ)
    print("目标：",ang1)
    output=ang1 * distinguish
    print("AB和CD的夹角")
    return ang1



def ds_ofpoint( a, b):
    x1, y1 = a
    x2, y2 = b
    distances = int(sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
    return distances

def remove_diff(deg):
    """
    :funtion :
    :param b:
    :param c:
    :return:
    """
    if (True):
        # new_nums = list(set(deg)) #剔除重复元素
        mean = np.mean(deg)
        var = np.var(deg)
        # print("原始数据共", len(deg), "个\n", deg)
        '''
        for i in range(len(deg)):
            print(deg[i],'→',(deg[i] - mean)/var)
            #另一个思路，先归一化，即标准正态化，再利用3σ原则剔除异常数据，反归一化即可还原数据
        '''
        # print("中位数:",np.median(deg))
        percentile = np.percentile(deg, (25, 50, 75), interpolation='midpoint')
        # print("分位数：", percentile)
        # 以下为箱线图的五个特征值
        Q1 = percentile[0]  # 上四分位数
        Q3 = percentile[2]  # 下四分位数
        IQR = Q3 - Q1  # 四分位距
        ulim = Q3 + 2.5 * IQR  # 上限 非异常范围内的最大值
        llim = Q1 - 1.5 * IQR  # 下限 非异常范围内的最小值

        new_deg = []
        uplim = []
        for i in range(len(deg)):
            if (llim < deg[i] and deg[i] < ulim):
                new_deg.append(deg[i])
        # print("清洗后数据共", len(new_deg), "个\n", new_deg)
    new_deg = np.mean(new_deg)

    return new_deg
    # 图表表达

def det_black( img):

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 0, 0])
    upper_hsv = np.array([180, 255, 110])
    mask = cv2.inRange(img2, lower_hsv, upper_hsv)
    imgResult = cv2.bitwise_and(img, img, mask=mask)


    return imgResult

def angle(v1, v2):
    x1, y1 = [v1[2] - v1[0], v1[3] - v1[1]]
    x2, y2 = [v2[2] - v2[0], v2[3] - v2[1]]
    # x1, y1 = [v1[2] - v1[0], v1[3] - v1[1]]
    # x2, y2 = [v2[0] - v2[0], v2[3] - v2[1]]
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    theta = np.arctan2(det, dot)
    theta = theta if theta > 0 else 2 * np.pi + theta
    class_meter = 0  # 0 表示 压力表， 1表示 -40到80的温度表,2表示-30到80的温度表
    if(class_meter == 0):
        distinguish = METER_distinguish['pressure_meter']  # 比例关系
        output =(360- (theta * 180 / np.pi) -45) * distinguish
        print("角度:__",360- (theta * 180 / np.pi) -45 )
    elif(class_meter == 1):
        distinguish = METER_distinguish['temp_meter_40']  # 比例关系
        output = (360 - (theta * 180 / np.pi) - 45) * distinguish - 40
        print("角度:__", 360 - (theta * 180 / np.pi) - 45)
    elif(class_meter == 2):
        distinguish = METER_distinguish['temp_meter_30']  # 比例关系
        output = (360 - (theta * 180 / np.pi) - 45) * distinguish - 30
        print("角度:__", 360 - (theta * 180 / np.pi) - 45)
    else:
        print("请选择相关的表通过class_meter变量。")
    return output

def cut_pic(img):
    """
    :param pyrMeanShiftFiltering(input, 10, 100) 均值滤波
    :param 霍夫概率圆检测
    :param mask操作提取圆
    :return: 半径，圆心位置

    """
    # input0 = cv2.imread(path)
    dst = cv2.pyrMeanShiftFiltering(img, 10, 100)

    cimage = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(cimage, cv2.HOUGH_GRADIENT, 1, 80, param1=100, param2=20, minRadius=80, maxRadius=0)
    circles = np.uint16(np.around(circles))  # 把类型换成整数
    r_1 = circles[0, 0, 2]
    c_x = circles[0, 0, 0]
    c_y = circles[0, 0, 1]
    circle = np.ones(img.shape, dtype="uint8")
    circle = circle * 255
    cv2.circle(circle, (c_x, c_y), int(r_1), 0, -1)

    ninfo = [r_1, c_x, c_y]
    return ninfo

def linecontours(cp_info,img):
    """
    :funtion : 提取刻度线，指针
    :param a: 高斯滤波 GaussianBlur，自适应二值化adaptiveThreshold，闭运算
    :param b: 轮廓寻找 findContours，
    :return:kb,new_needleset
    """
    r_1, c_x, c_y = cp_info

    cv2.circle(img, (c_x, c_y), 20, (23, 28, 28), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(~gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10)

    contours, hier = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cntset = []  # 刻度线轮廓集合
    cntareas = []  # 刻度线面积集合
    needlecnt = []  # 指针轮廓集合
    needleareas = []  # 指针面积集合
    ca = (c_x, c_y)
    incircle = [r_1 * 0.7, r_1 * 0.9]

    localtion = []
    for xx in contours:
        rect = cv2.minAreaRect(xx)
        # print(rect)
        a, b, c = rect
        w, h = b
        w = int(w)
        h = int(h)
        ''' 满足条件:“长宽比例”，“面积”'''
        if h == 0 or w == 0:
            pass
        else:
            dis = ds_ofpoint( a=ca, b=a)

            if (incircle[0] < dis and incircle[1] > dis):
                localtion.append(dis)
                if h / w > 4 or w / h > 4:
                    cntset.append(xx)
                    cntareas.append(w * h)
            else:
                if w > r_1 / 2 or h > r_1 / 2:
                    needlecnt.append(xx)
                    needleareas.append(w * h)
    cntareas = np.array(cntareas)
    nss = remove_diff(cntareas)  # 中位数，上限区
    new_cntset = []
    # 面积
    for i, xx in enumerate(cntset):
        if (cntareas[i] <= nss * 50 and cntareas[i] >= nss * 0.5):
            new_cntset.append(xx)
    kb = []  # 拟合线集合
    ks = []
    k_add = 0
    k_i = 0
    # print('new_cntset',new_cntset)
    for xx in new_cntset:
        rect = cv2.minAreaRect(xx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.polylines(img, [box], True, (0, 255, 0), 1)  # pic
        output = cv2.fitLine(xx, 2, 0, 0.001, 0.001)
        # print("output",output)

        k = output[1] / output[0]
        k = round(k[0], 2)
        b = output[3] - k * output[2]
        b = round(b[0], 2)
        x1 = 1
        x2 = gray.shape[0]
        y1 = int(k * x1 + b)
        y2 = int(k * x2 + b)
        # print("x,y",x1,y1,x2,y2)
        # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        kb.append([k, b])  # 求中心点的点集[k,b]
        k_add += k
        k_i +=1
        ks.append(k)
        # cv2.putText(img,str(np.size(kb)),(x2, y2),cv2.FONT_HERSHEY_SIMPLEX,
        #             0.3, (0, 0, 0), 1, cv2.LINE_AA)


    ############################################################
    r = np.mean(localtion)
    k_means =  k_add/k_i
    # remove_num_diff(ks)
    print("k_means",k_means,"\n k size",np.size(ks))
    mask = np.zeros(img.shape[0:2], np.uint8)

    mask = cv2.drawContours(mask, needlecnt, -1, (255, 255, 255), -1)  # 生成掩膜

    return kb, r, mask

def needle(img, r, cx, cy):
    oimg =img.copy()

    # circle = np.ones(img.shape, dtype="uint8")
    # circle = circle * 255
    circle = np.zeros(img.shape, dtype="uint8")
    cv2.circle(circle, (cx, cy), int(r), 255, -1)
    mask = cv2.bitwise_and(img, circle)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)


    lines = cv2.HoughLinesP(mask, 1, np.pi / 180, 1, maxLineGap=2)
    nmask = np.zeros(img.shape, np.uint8)
    # lines = mential.findline(self=0, cp=[x, y], lines=lines)
    # print('lens', len(lines))
    newNask = nmask
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # cv2.line(newNask, (x1, y1), (x2, y2), (255,255,255), 2, cv2.LINE_AA)
        cv2.line(nmask, (x1, y1), (x2, y2), 100, 1, cv2.LINE_AA)
    x1, y1, x2, y2 = lines[0][0]
    d1 = (x1 - cx) ** 2 + (y1 - cy) ** 2
    d2 = (x2 - cx) ** 2 + (y2 - cy) ** 2
    if d1 > d2:
        axit = [x1, y1]
    else:
        axit = [x2, y2]
    nmask = cv2.erode(nmask, kernel, iterations=1)

    cnts, hier = cv2.findContours(nmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts, hier = cv2.findContours(nmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    areass = [cv2.contourArea(x) for x in cnts]
    # print(len(areass))
    i = areass.index(max(areass))


    cnt = cnts[i]
    output = cv2.fitLine(cnt, 2, 0, 0.001, 0.001)
    k = output[1] / output[0]
    k = round(k[0], 2)
    b = output[3] - k * output[2]
    b = round(b[0], 2)
    x1 = cx
    x2 = axit[0]
    y1 = int(k * x1 + b)

    y2 = int(k * x2 + b)
    cv2.line(oimg, (x1, y1), (x2, y2), (0, 23, 255), 1, cv2.LINE_AA)
    cv2.line(oimg, (x1, y1), (cx,cy), (0, 23, 255), 1, cv2.LINE_AA)


    # print(pname +'_fin'+ ptype)
    # cv2.imwrite(pname +'_fin'+ ptype,oimg)
    return x1, y1, x2, y2

def decter(imgpath):
    # x0=x
    # y0=y
    global  pname, ptype
    nor_img = resize(imgpath,METER_SHAPE)
    ninfo = cut_pic(nor_img)  # 2.截取表盘
    kb, r, mask = linecontours(ninfo,nor_img)

    cx, cy = CIRCLE_CENTER[0],CIRCLE_CENTER[1]
    print("cx,cy",cx,cy)
    da, db, dc, de = needle(mask, r, cx, cy)
    print(da,db,dc,de)
    distinguish = 100 / 360
    OZ = [da, db, cx, cy]
    OP = [da, db, dc, de]
    dab = (cx-da)*(cx-da) +(cy-db)*(cy-db)
    dce = (cx-dc)*(cx-dc) +(cy-de)*(cy-de)
    print("差距：",dab ,dce)
    if(dab< dce):
        OZ = [da, db, da, 300]
        OP = [da, db, dc, de]
    cv2.line(nor_img, (da, db), (da, 300), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(nor_img, (da, db), (dc, de), (0, 255, 0), 2, cv2.LINE_AA)
    ang1 = angle(OP, OZ)
    print("目标：",ang1)
    output=ang1 * distinguish
    print("AB和CD的夹角")
    return output
# TODO 截止到此，以上是角度


##########################################################################


def resize(imgs, target_size, interp=cv2.INTER_LINEAR):
    """图像缩放至固定大小

    参数：
        imgs (list[np.array])：批量BGR图像数组。
        target_size (list|tuple)：缩放后的图像大小，格式为[高, 宽]。
        interp (int)：图像差值方法。默认值为cv2.INTER_LINEAR。

    返回：
        resized_imgs (list[np.array])：缩放后的批量BGR图像数组。
    """
    img_shape = imgs.shape
    scale_x = float(target_size[1]) / float(img_shape[1])
    scale_y = float(target_size[0]) / float(img_shape[0])
    resize_img = cv2.resize(
        imgs, None, None, fx=scale_x, fy=scale_y, interpolation=interp)
    return resize_img
def erode(seg_results, erode_kernel):
    """对分割模型预测结果中label_map做图像腐蚀操作

    参数：
        seg_results (list[dict])：分割模型的预测结果。
        erode_kernel (int): 图像腐蚀的卷积核的大小。

    返回：
        eroded_results (list[dict])：对label_map进行腐蚀后的分割模型预测结果。

    """
    kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
    eroded_results = seg_results
    # for i in range(len(seg_results)):
    #     test_resulte = seg_results[i]['label_map']
        # print('***********************************',type(test_resulte))
        # eroded_results[i]['label_map'] = cv2.erode(
        #     seg_results[i]['label_map'], kernel)
        # eroded_results[i]['label_map'] = cv2.erode(
        #     test_resulte.astype('uint8'), kernel)
    eroded_results= cv2.erode(
        eroded_results.astype('uint8'), kernel)
    return eroded_results


def circle_to_rectangle(seg_results):
    """将圆形表盘的预测结果label_map转换成矩形

    圆形到矩形的计算方法：
        因本案例中两种表盘的刻度起始值都在左下方，故以圆形的中心点为坐标原点，
        从-y轴开始逆时针计算极坐标到x-y坐标的对应关系：
          x = r + r * cos(theta)
          y = r - r * sin(theta)
        注意：
            1. 因为是从-y轴开始逆时针计算，所以r * sin(theta)前有负号。
            2. 还是因为从-y轴开始逆时针计算，所以矩形从上往下对应圆形从外到内，
               可以想象把圆形从-y轴切开再往左右拉平时，圆形的外围是上面，內围在下面。

    参数：
        seg_results (list[dict])：分割模型的预测结果。

    返回值：
        rectangle_meters (list[np.array])：矩形表盘的预测结果label_map。

    """
    # rectangle_meters = list()
    # for i, seg_result in enumerate(seg_results):
    #     label_map = seg_result['label_map']
    #     # rectangle_meter的大小已经由预先设置的全局变量RECTANGLE_HEIGHT, RECTANGLE_WIDTH决定
    #     rectangle_meter = np.zeros(
    #         (RECTANGLE_HEIGHT, RECTANGLE_WIDTH), dtype=np.uint8)
    #     for row in range(RECTANGLE_HEIGHT):
    #         for col in range(RECTANGLE_WIDTH):
    #             theta = PI * 2 * (col + 1) / RECTANGLE_WIDTH
    #             # 矩形从上往下对应圆形从外到内
    #             rho = CIRCLE_RADIUS - row - 1
    #             y = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.5)
    #             x = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.5)
    #             rectangle_meter[row, col] = label_map[y, x]
    #     rectangle_meters.append(rectangle_meter)

    rectangle_meters = list()

    rectangle_meter = np.zeros(
        (RECTANGLE_HEIGHT, RECTANGLE_WIDTH,3), dtype=np.uint8)
    for row in range(RECTANGLE_HEIGHT):

        for col in range(RECTANGLE_WIDTH):
            theta = PI * 2 * (col + 1) / RECTANGLE_WIDTH
            # 矩形从上往下对应圆形从外到内
            rho = CIRCLE_RADIUS - row - 1
            y = int(CIRCLE_CENTER[0] + rho * math.cos(theta) + 0.5)
            x = int(CIRCLE_CENTER[1] - rho * math.sin(theta) + 0.5)

            rectangle_meter[row, col] = seg_results[y, x]
            black_img_gray = cv2.cvtColor(rectangle_meter, cv2.COLOR_BGR2GRAY)


    return black_img_gray

def rectangle_to_line( rectangle_meter):
    """从矩形表盘的预测结果中提取指针和刻度预测结果并沿高度方向压缩成线状格式。

    参数：
        rectangle_meters (list[np.array])：矩形表盘的预测结果label_map。

    返回：
        line_scales (list[np.array])：刻度的线状预测结果。
        line_pointers (list[np.array])：指针的线状预测结果。

    """
    line_scales = list()
    line_pointers = list()

    height, width = rectangle_meter.shape[0:2]
    line_scale = np.zeros((width), dtype=np.uint8)
    line_pointer = np.zeros((width), dtype=np.uint8)
    print("hHHHH",height,"WWWWW",width)

    for col in range(width):
        for row in range(height):
            if rectangle_meter[row, col] == SEG_CNAME2CLSID['pointer']:
                line_pointer[col] += 1
            elif rectangle_meter[row, col] == SEG_CNAME2CLSID['scale']:
                line_scale[col] += 1

    line_scales.append(line_scale)
    line_pointers.append(line_pointer)
    return line_scales, line_pointers

def rectangle_to_lines1(imgsca_re ,imgpoi_re):
    """从矩形表盘的预测结果中提取指针和刻度预测结果并沿高度方向压缩成线状格式。

    参数：
        rectangle_meters (list[np.array])：矩形表盘的预测结果label_map。

    返回：
        line_scales (list[np.array])：刻度的线状预测结果。
        line_pointers (list[np.array])：指针的线状预测结果。

    """
    line_scales = list()
    line_pointers = list()

    height, width = imgsca_re.shape[0:2]
    line_scale = np.zeros((width), dtype=np.uint8)
    line_pointer = np.zeros((width), dtype=np.uint8)

    for col in range(width):
        for row in range(height):
            if imgpoi_re[row, col] == SEG_CNAME2CLSID['pointer']:
                line_pointer[col] += 1

            if imgsca_re[row, col] == SEG_CNAME2CLSID['scale']:
                line_scale[col] += 1

    line_scales.append(line_scale)
    line_pointers.append(line_pointer)
    return line_scales, line_pointers

def mean_binarization( data_list):
    """对图像进行均值二值化操作

    参数：
        data_list (list[np.array])：待二值化的批量数组。

    返回：
        binaried_data_list (list[np.array])：二值化后的批量数组。

    """
    batch_size = len(data_list)
    binaried_data_list = data_list
    for i in range(batch_size):
        mean_data = np.mean(data_list[i])
        print("mean_data",mean_data)
        width = data_list[i].shape[0]
        for col in range(width):
            if data_list[i][col] < mean_data:
                binaried_data_list[i][col] = 0
            else:
                binaried_data_list[i][col] = 1
    return binaried_data_list

def locate_scale(line_scales):
    """在线状预测结果中找到每根刻度的中心位置

    参数：
        line_scales (list[np.array])：批量的二值化后的刻度线状预测结果。

    返回：
        scale_locations (list[list])：各图像中每根刻度的中心位置。

    """
    batch_size = len(line_scales)
    scale_locations = list()
    for i in range(batch_size):
        line_scale = line_scales[i]
        width = line_scale.shape[0]
        find_start = False
        one_scale_start = 0
        one_scale_end = 0
        locations = list()
        for j in range(width - 1):
            # print("line_scale::::",line_scale[j])
            if line_scale[j] > 0 :
                if find_start == False:
                    one_scale_start = j
                    find_start = True
            if find_start:
                if line_scale[j] == 0 and line_scale[j + 1] == 0:
                    one_scale_end = j - 1
                    one_scale_location = (
                        one_scale_start + one_scale_end) / 2
                    locations.append(one_scale_location)
                    one_scale_start = 0
                    one_scale_end = 0
                    find_start = False
        scale_locations.append(locations)
    return scale_locations
def locate_pointer(line_pointers):
    """在线状预测结果中找到指针的中心位置

    参数：
        line_scales (list[np.array])：批量的指针线状预测结果。

    返回：
        scale_locations (list[list])：各图像中指针的中心位置。

    """
    batch_size = len(line_pointers)
    pointer_locations = list()
    for i in range(batch_size):
        line_pointer = line_pointers[i]
        find_start = False
        pointer_start = 0
        pointer_end = 0
        location = 0
        width = line_pointer.shape[0]
        for j in range(width - 3):
            if line_pointer[j] > 0 and line_pointer[j + 1] > 0 and line_pointer[j + 2] > 0 and line_pointer[j + 3] > 0:
                if find_start == False:
                    pointer_start = j
                    find_start = True
            if find_start:
                if line_pointer[j] == 0 and line_pointer[j + 1] == 0:
                    pointer_end = j - 1
                    location = (pointer_start + pointer_end) / 2
                    find_start = False
                    break
        pointer_locations.append(location)
    return pointer_locations

def get_relative_location(scale_locations, pointer_locations):
    """找到指针指向了第几根刻度

    参数：
        scale_locations (list[list])：批量的每根刻度的中心点位置。
        pointer_locations (list[list])：批量的指针的中心点位置。

    返回：
        pointed_scales (list[dict])：每个表的结果组成的list。每个表的结果由字典表示，
            字典有两个关键词：'num_scales'、'pointed_scale'，分别表示预测的刻度根数、
            预测的指针指向了第几根刻度。

    """

    pointed_scales = list()
    for scale_location, pointer_location in zip(scale_locations,
                                                pointer_locations):
        num_scales = len(scale_location)
        pointed_scale = -1
        if num_scales > 0:
            for i in range(num_scales - 1):
                if scale_location[
                        i] <= pointer_location and pointer_location < scale_location[
                            i + 1]:
                    pointed_scale = i + (
                        pointer_location - scale_location[i]
                    ) / (scale_location[i + 1] - scale_location[i] + 1e-05
                         ) + 1
        result = {'num_scales': num_scales, 'pointed_scale': pointed_scale}
        pointed_scales.append(result)
    return pointed_scales

def calculate_reading( pointed_scales):
    """根据刻度的间隔值和指针指向的刻度根数计算表盘的读数
    """
    readings = list()
    batch_size = len(pointed_scales)
    for i in range(batch_size):
        pointed_scale = pointed_scales[i]
        # 刻度根数大于阈值的为第一种表盘
        if pointed_scale['num_scales'] > TYPE_THRESHOLD[1]:
            reading = (pointed_scale['pointed_scale']) * METER_CONFIG[0][
                'scale_interval_value'] - 30
        elif  pointed_scale['num_scales'] > TYPE_THRESHOLD[0] :
            reading = pointed_scale['pointed_scale'] * METER_CONFIG[1][
                'scale_interval_value']
        else:
            reading = pointed_scale['pointed_scale'] * METER_CONFIG[2][
                'scale_interval_value'] - 20


        if(reading < 0 ):
            reading = decter_only_line('2.png')  # TODO 输入分割后的指针
            print("laizheli")

        readings.append(reading)

    return readings

def print_meter_readings(meter_readings):
    """打印各表盘的读数

    参数：
        meter_readings (list[dict])：各表盘的读数
    """
    mess = []
    for i in range(len(meter_readings)):
        if (meter_readings[i] == -22.0):
            print("")
        else:
            print("表盘的度数为 {}: {}".format(i + 1, meter_readings[i]))
            mess.append("表盘的度数为 {}: {}".format(i + 1, meter_readings[i]))

    return mess




def read_meter(image_scale,image_point):
    imgsca = cv2.imread(image_scale)
    imgpoi = cv2.imread(image_point)
    imgsca = resize(imgsca,METER_SHAPE)
    imgpoi =  resize(imgpoi,METER_SHAPE)
    imgsca_erode = erode(imgsca,1)
    imgpoi_erode = erode(imgpoi,1)
    # cv2.imshow("imgsca_erode",imgsca_erode)
    # cv2.imshow("imgpoi_erode", imgpoi_erode)
    # cv2.waitKey(0)
    imgsca_re = circle_to_rectangle(imgsca_erode)
    imgpoi_re = circle_to_rectangle(imgpoi_erode)

    # cv2.imwrite("data/imgsca_re.png",imgsca_re)
    # cv2.imwrite("data/imgpoi_re.png",imgpoi_re)

    line_scales, line_pointers = rectangle_to_lines1(imgsca_re,imgpoi_re)
    binaried_scales = mean_binarization(line_scales)
    binaried_pointers = mean_binarization(line_pointers)

    scale_locations = locate_scale(binaried_scales)
    pointer_locations =locate_pointer(binaried_pointers)

    pointed_scales = get_relative_location(scale_locations,
                                                pointer_locations)
    # print("pointed_scales",pointed_scales)
    meter_readings = calculate_reading(pointed_scales)

    return print_meter_readings(meter_readings)





class Tester(object):

    def __init__(self, is_cuda=False):
        self.net = U2NET(3, 2)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and is_cuda else 'cpu')
        self.net.load_state_dict(torch.load(r'E:\PyProject\ReadMeter(1)\ReadMeter\weight\net4.pt', map_location='cpu'))
        self.net.eval().to(self.device)


    @torch.no_grad()
    def __call__(self, save_path=r'E:\PyProject\ReadMeter(1)\ReadMeter\2', image_name='1', image=r'E:\PyProject\ReadMeter(1)\ReadMeter\2\2222..jpg'):
        image=cv2.imread(image)
        image = self.square_picture(image, 512)
        image_tensor = self.to_tensor(image.copy()).to(self.device)
        d0, d1, d2, d3, d4, d5, d6 = self.net(image_tensor)
        mask = d0.squeeze(0).cpu().numpy()
        # cv2.imshow('mask[0]', mask[0])
        # cv2.imshow('mask[1]', mask[1])
        # cv2.waitKey(0)

        dail_mask = self.binary_image(mask[0])
        point_mask = self.binary_image(mask[1])
        # cv2.imshow('point_mask', point_mask)
        # cv2.imshow('dail_mask', dail_mask)
        # print(point_mask.dtype)
        # print(dail_mask.dtype)
        # cv2.waitKey(0)
        cv2.imwrite("1.png",dail_mask*255)
        cv2.imwrite("2.png", point_mask*255)
        save_message = read_meter("1.png","2.png")  #########################
        # ang1 = decter_only_line('2.png')  # TODO 输入分割后的指针
        # ang1 = decter(image)
        # print("表盘刻度为：",ang1)
        # ang2=str(ang1)
        # print(save_path, image_name)
        # print(str(save_path) + '/' + str(image_name) + '.txt')
        for item in save_message:
            with open(str(save_path) + '/' + str(image_name) + '.txt', "a") as f:
                f.write(item)

        # with open(str(save_path) + '/' + str(image_name) + '.txt', "a") as f:
        #     f.write("表盘刻度为 1: ")
        #     for i in range(len(ang2)):
        #         f.write(ang2[i])
        #     f.write("\n")
        #     f.close()

        cv2.imshow('image', image)

        # condition = point_mask == 1
        # image[condition] = (0, 0, 255)
        # condition = dail_mask == 1
        # image[condition] = (0, 255, 0)xwz_state does not exist
        cv2.waitKey(10)
        pass

    def corrosion(self, image):
        """
        腐蚀操作
        :param image:
        :return:
        """
        kernel = numpy.ones((3, 3), numpy.uint8)
        image = cv2.erode(image, kernel)
        return image

    def binary_image(self, image):
        condition = image > 0.5
        image[condition] = 1
        image[~condition] = 0
        image = self.corrosion(image)
        return image

    @staticmethod
    def to_tensor(image):
        image = torch.tensor(image).float() / 255
        image = image.permute(2, 0, 1).unsqueeze(0)
        return image

    @staticmethod
    def square_picture(image, image_size):
        """
        任意图片正方形中心化
        :param image: 图片
        :param image_size: 输出图片的尺寸
        :return: 输出图片
        """
        h1, w1, _ = image.shape
        max_len = max(h1, w1)
        fx = image_size / max_len
        fy = image_size / max_len
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
        h2, w2, _ = image.shape
        background = numpy.zeros((image_size, image_size, 3), dtype=numpy.uint8)
        background[:, :, :] = 127
        s_h = image_size // 2 - h2 // 2
        s_w = image_size // 2 - w2 // 2
        background[s_h:s_h + h2, s_w:s_w + w2] = image
        return background


tester = Tester()
tester()
