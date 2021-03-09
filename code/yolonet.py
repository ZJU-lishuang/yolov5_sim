import torch
import torch.nn as nn
import math

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))

class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)

class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.stride = None  # strides computed during build
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.export = False  # onnx export

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



class Yolov5(nn.Module):
    def __init__(self,gd=1,gw=1,nc=80,anchor=[[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]):
        super(Yolov5, self).__init__()
        self.gd=gd
        self.gw=gw
        divisor=8
        self.nc=nc
        self.anchor=anchor
        width_n=lambda x:math.ceil(x*self.gw / divisor) * divisor
        depth_n=lambda x:max(round(x * self.gd), 1) if x > 1 else x
        self.focus=Focus(3,width_n(64),3)
        self.conv1=Conv(width_n(64), width_n(128), 3, 2)
        self.BottleneckCSP1=BottleneckCSP(width_n(128), width_n(128),depth_n(3))
        self.conv2 = Conv(width_n(128), width_n(256), 3, 2)
        self.BottleneckCSP2 = BottleneckCSP(width_n(256), width_n(256), depth_n(9))
        self.conv3 = Conv(width_n(256), width_n(512), 3, 2)
        self.BottleneckCSP3 = BottleneckCSP(width_n(512), width_n(512), depth_n(9))
        self.conv4 = Conv(width_n(512), width_n(1024), 3, 2)
        self.spp=SPP(width_n(1024), width_n(1024),[5,9,13])
        self.BottleneckCSP4 = BottleneckCSP(width_n(1024), width_n(1024), depth_n(3),False)

        self.conv5=Conv(width_n(1024), width_n(512), 1, 1)
        self.Upsample1=nn.Upsample(None,2,'nearest')
        self.concat1=Concat(1)
        self.BottleneckCSP5 = BottleneckCSP(width_n(1024), width_n(512), depth_n(3),False)

        self.conv6 = Conv(width_n(512), width_n(256), 1, 1)
        self.Upsample2 = nn.Upsample(None, 2, 'nearest')
        self.concat2 = Concat(1)
        self.BottleneckCSP6 = BottleneckCSP(width_n(512), width_n(256), depth_n(3), False)

        self.conv7 = Conv(width_n(256), width_n(256), 3, 2)
        self.concat3 = Concat(1)
        self.BottleneckCSP7 = BottleneckCSP(width_n(512), width_n(512), depth_n(3), False)

        self.conv8 = Conv(width_n(512), width_n(512), 3, 2)
        self.concat4 = Concat(1)
        self.BottleneckCSP8 = BottleneckCSP(width_n(1024), width_n(1024), depth_n(3), False)

        self.Detect=Detect(self.nc,self.anchor,[width_n(256), width_n(512), width_n(1024)])

    def forward(self,x):
        x=self.focus(x)
        x=self.conv1(x)
        x=self.BottleneckCSP1(x)
        x = self.conv2(x)
        x2 = self.BottleneckCSP2(x)
        x = self.conv3(x2)
        x1 = self.BottleneckCSP3(x)
        x = self.conv4(x1)
        x=self.spp(x)
        x = self.BottleneckCSP4(x)

        x4=self.conv5(x)
        x=self.Upsample1(x4)
        x=self.concat1([x,x1])
        x=self.BottleneckCSP5(x)

        x3=self.conv6(x)
        x=self.Upsample2(x3)
        x=self.concat2([x,x2])
        x5=self.BottleneckCSP6(x)

        x=self.conv7(x5)
        x=self.concat3([x,x3])
        x6=self.BottleneckCSP7(x)

        x=self.conv8(x6)
        x=self.concat4([x,x4])
        x7=self.BottleneckCSP8(x)

        x=self.Detect([x5,x6,x7])

        return x

def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

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

if __name__ == '__main__':
    weights="yolov5l.pt"
    model = torch.load(weights, map_location=torch.device('cpu'))['model'].float()

    numclass = model.nc
    anchor = model.yaml['anchors']
    gd=model.yaml['depth_multiple']
    gw=model.yaml['width_multiple']
    #初始化模型
    yolomodel = Yolov5(gd=gd,gw=gw,nc=numclass,anchor=anchor)

    initialize_weights(yolomodel)

    m = yolomodel.Detect
    s = 128  # 2x min stride
    ch=3
    m.stride = torch.tensor([s / x.shape[-2] for x in yolomodel.forward(torch.zeros(1, ch, s, s))])  # forward
    m.anchors /= m.stride.view(-1, 1, 1)
    check_anchor_order(m)

    #初始化参数
    yolo_state_dict=model.state_dict()
    state_dict=yolomodel.state_dict()

    weights = {}
    for [k1,v1],[k2,v2] in zip(yolo_state_dict.items(),state_dict.items()):
        # print("k1=",k1)
        # print("k2=",k2)
        assert k1.split('.', 2)[2]==k2.split('.', 1)[1]
        weights[k2]=v1
    yolomodel.load_state_dict(weights)

    #模型测试
    yolomodel.eval()
    model.eval()
    device='cuda:0'
    model = model.to(device)
    yolomodel = yolomodel.to(device)
    # device = 'cpu'
    img = torch.rand(1 if torch.cuda.is_available() else 1, 3, 640, 640).to(device)

    y1 = model(img)
    y2 = yolomodel(img)

    assert (y1[0] == y2[0]).all()
