# from utils.datasets import *
# from utils.utils import *
import torch
import cv2
import numpy as np
import time
import random

from pre import letterbox
from post import non_max_suppression,xyxy2xywh,scale_coords

from yolonet import Yolov5
import torch.nn as nn

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            # m.momentum = 0.03
            m.momentum = 0.01
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True

def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

def detect_test():
    weights = '../model/best_l416320.pt'
    path = '../sample/images/000159.jpg'
    save_path = '../sample/results/000159.jpg'
    cuda=True
    device=torch.device('cuda:0' if cuda else 'cpu')
    conf_thres=0.4
    iou_thres=0.5
    classes=None
    # model=torch.load(weights, map_location=device)['model'].float().fuse().eval()
    model = torch.load(weights, map_location=device)['model'].float().eval()

    names = model.module.names if hasattr(model, 'module') else model.names
    numclass=model.nc
    anchor=model.yaml['anchors']
    gd = model.yaml['depth_multiple']
    gw = model.yaml['width_multiple']

    yolomodel = Yolov5(gd=gd, gw=gw, nc=numclass, anchor=anchor)
    initialize_weights(yolomodel)
    m = yolomodel.Detect
    s = 128  # 2x min stride
    ch = 3
    m.stride = torch.tensor([s / x.shape[-2] for x in yolomodel.forward(torch.zeros(1, ch, s, s))])  # forward
    m.anchors /= m.stride.view(-1, 1, 1)
    check_anchor_order(m)

    yolo_state_dict = model.state_dict()
    state_dict = yolomodel.state_dict()

    weights = {}
    for [k1, v1], [k2, v2] in zip(yolo_state_dict.items(), state_dict.items()):
        # print("k1=",k1)
        # print("k2=",k2)
        assert k1.split('.', 2)[2] == k2.split('.', 1)[1]
        weights[k2] = v1
    yolomodel.load_state_dict(weights)
    yolomodel = yolomodel.to(device).eval()



    # half = device.type != 'cpu'
    # if half:
    #     model.half()  # to FP16

    img0 = cv2.imread(path)  # BGR
    img = letterbox(img0, new_shape=512)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    t1 = time_synchronized()
    # pred = model(img)[0]
    pred = yolomodel(img)[0]

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes)
    t2 = time_synchronized()
    print('Done. (%.3fs)' % ( t2 - t1))

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        p= path
        im0=img0

        tl = round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in det:
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                x0, y0, w0, h0 = xywh
                h, w = im0.shape[:2]
                x0 *= w
                y0 *= h
                w0 *= w
                h0 *= h
                x1 = x0 - w0 / 2
                y1 = y0 - h0 / 2
                print((('%s ' + '%.2g ' + '%d ' * 3 + '%d' + '\n') % (
                        names[int(cls)], conf, x1, y1, w0, h0)))
                color = [random.randint(0, 255) for _ in range(3)]
                c1, c2 = (int(x1), int(y1)), (int(x1+w0), int(y1+h0))
                cv2.rectangle(im0, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        cv2.imwrite(save_path, im0)




if __name__ == '__main__':
    detect_test()
