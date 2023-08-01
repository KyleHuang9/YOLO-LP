import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, dist2cor


class Detect(nn.Module):
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, npro=31, nalp=24, nads=37, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.npro = npro  # number of provinces
        self.nalp = nalp  # number of alphabets
        self.nads = nads  # number of charactors
        self.no = npro + nalp + nads * 5 + 13  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64] # strides computed during build
        self.stride = torch.tensor(stride)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.pro_preds = nn.ModuleList()
        self.alp_preds = nn.ModuleList()
        self.ad0_preds = nn.ModuleList()
        self.ad1_preds = nn.ModuleList()
        self.ad2_preds = nn.ModuleList()
        self.ad3_preds = nn.ModuleList()
        self.ad4_preds = nn.ModuleList()
        self.ad5_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cor_preds = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i*13
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.pro_preds.append(head_layers[idx+3])
            self.alp_preds.append(head_layers[idx+4])
            self.ad0_preds.append(head_layers[idx+5])
            self.ad1_preds.append(head_layers[idx+6])
            self.ad2_preds.append(head_layers[idx+7])
            self.ad3_preds.append(head_layers[idx+8])
            self.ad4_preds.append(head_layers[idx+9])
            self.ad5_preds.append(head_layers[idx+10])
            self.reg_preds.append(head_layers[idx+11])
            self.cor_preds.append(head_layers[idx+12])

    def initialize_biases(self):

        for conv in self.pro_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        for conv in self.alp_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        for conv in self.ad0_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        for conv in self.ad1_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        for conv in self.ad2_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        for conv in self.ad3_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        for conv in self.ad4_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        for conv in self.ad5_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        for conv in self.cor_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x):
        if self.training:
            pro_score_list = []
            alp_score_list = []
            ad0_score_list = []
            ad1_score_list = []
            ad2_score_list = []
            ad3_score_list = []
            ad4_score_list = []
            ad5_score_list = []
            reg_distri_list = []
            cor_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                pro_output = self.pro_preds[i](cls_feat)
                alp_output = self.alp_preds[i](cls_feat)
                ad0_output = self.ad0_preds[i](cls_feat)
                ad1_output = self.ad1_preds[i](cls_feat)
                ad2_output = self.ad2_preds[i](cls_feat)
                ad3_output = self.ad3_preds[i](cls_feat)
                ad4_output = self.ad4_preds[i](cls_feat)
                ad5_output = self.ad5_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                cor_output = self.cor_preds[i](reg_feat)

                pro_output = torch.sigmoid(pro_output)
                alp_output = torch.sigmoid(alp_output)
                ad0_output = torch.sigmoid(ad0_output)
                ad1_output = torch.sigmoid(ad1_output)
                ad2_output = torch.sigmoid(ad2_output)
                ad3_output = torch.sigmoid(ad3_output)
                ad4_output = torch.sigmoid(ad4_output)
                ad5_output = torch.sigmoid(ad5_output)

                pro_score_list.append(pro_output.flatten(2).permute((0, 2, 1)))
                alp_score_list.append(alp_output.flatten(2).permute((0, 2, 1)))
                ad0_score_list.append(ad0_output.flatten(2).permute((0, 2, 1)))
                ad1_score_list.append(ad1_output.flatten(2).permute((0, 2, 1)))
                ad2_score_list.append(ad2_output.flatten(2).permute((0, 2, 1)))
                ad3_score_list.append(ad3_output.flatten(2).permute((0, 2, 1)))
                ad4_score_list.append(ad4_output.flatten(2).permute((0, 2, 1)))
                ad5_score_list.append(ad5_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))
                cor_distri_list.append(cor_output.flatten(2).permute((0, 2, 1)))

            pro_score_list = torch.cat(pro_score_list, axis=1)
            alp_score_list = torch.cat(alp_score_list, axis=1)
            ad0_score_list = torch.cat(ad0_score_list, axis=1)
            ad1_score_list = torch.cat(ad1_score_list, axis=1)
            ad2_score_list = torch.cat(ad2_score_list, axis=1)
            ad3_score_list = torch.cat(ad3_score_list, axis=1)
            ad4_score_list = torch.cat(ad4_score_list, axis=1)
            ad5_score_list = torch.cat(ad5_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)
            cor_distri_list = torch.cat(cor_distri_list, axis=1)

            return x, pro_score_list, alp_score_list, ad0_score_list, ad1_score_list, ad2_score_list, ad3_score_list, ad4_score_list, ad5_score_list, reg_distri_list, cor_distri_list
        else:
            pro_score_list = []
            alp_score_list = []
            ad0_score_list = []
            ad1_score_list = []
            ad2_score_list = []
            ad3_score_list = []
            ad4_score_list = []
            ad5_score_list = []
            reg_dist_list = []
            cor_dist_list = []
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                pro_output = self.pro_preds[i](cls_feat)
                alp_output = self.alp_preds[i](cls_feat)
                ad0_output = self.ad0_preds[i](cls_feat)
                ad1_output = self.ad1_preds[i](cls_feat)
                ad2_output = self.ad2_preds[i](cls_feat)
                ad3_output = self.ad3_preds[i](cls_feat)
                ad4_output = self.ad4_preds[i](cls_feat)
                ad5_output = self.ad5_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                cor_output = self.cor_preds[i](reg_feat)

                if self.use_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_conv(F.softmax(reg_output, dim=1))

                pro_output = torch.sigmoid(pro_output)
                alp_output = torch.sigmoid(alp_output)
                ad0_output = torch.sigmoid(ad0_output)
                ad1_output = torch.sigmoid(ad1_output)
                ad2_output = torch.sigmoid(ad2_output)
                ad3_output = torch.sigmoid(ad3_output)
                ad4_output = torch.sigmoid(ad4_output)
                ad5_output = torch.sigmoid(ad5_output)

                pro_score_list.append(pro_output.reshape([b, self.npro, l]))
                alp_score_list.append(alp_output.reshape([b, self.nalp, l]))
                ad0_score_list.append(ad0_output.reshape([b, self.nads, l]))
                ad1_score_list.append(ad1_output.reshape([b, self.nads, l]))
                ad2_score_list.append(ad2_output.reshape([b, self.nads, l]))
                ad3_score_list.append(ad3_output.reshape([b, self.nads, l]))
                ad4_score_list.append(ad4_output.reshape([b, self.nads, l]))
                ad5_score_list.append(ad5_output.reshape([b, self.nads, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))
                cor_dist_list.append(cor_output.reshape([b, 8, l]))

            pro_score_list = torch.cat(pro_score_list, axis=-1).permute(0, 2, 1)
            alp_score_list = torch.cat(alp_score_list, axis=-1).permute(0, 2, 1)
            ad0_score_list = torch.cat(ad0_score_list, axis=-1).permute(0, 2, 1)
            ad1_score_list = torch.cat(ad1_score_list, axis=-1).permute(0, 2, 1)
            ad2_score_list = torch.cat(ad2_score_list, axis=-1).permute(0, 2, 1)
            ad3_score_list = torch.cat(ad3_score_list, axis=-1).permute(0, 2, 1)
            ad4_score_list = torch.cat(ad4_score_list, axis=-1).permute(0, 2, 1)
            ad5_score_list = torch.cat(ad5_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)
            cor_dist_list = torch.cat(cor_dist_list, axis=-1).permute(0, 2, 1)


            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_corners = dist2cor(cor_dist_list, anchor_points)
            pred_bboxes *= stride_tensor
            pred_corners *= stride_tensor
            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    pred_corners,
                    pro_score_list,
                    alp_score_list,
                    ad0_score_list,
                    ad1_score_list,
                    ad2_score_list,
                    ad3_score_list,
                    ad4_score_list,
                    ad5_score_list
                ],
                axis=-1)


def build_effidehead_layer(channels_list, num_anchors, npro, nalp, nads, reg_max=16, num_layers=3):

    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]

    head_layers = nn.Sequential(
        # stem0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # pro_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=npro * num_anchors,
            kernel_size=1
        ),
        # alp_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=nalp * num_anchors,
            kernel_size=1
        ),
        # ads0_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads1_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads2_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads3_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads4_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads5_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # cor_pred0
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=8 * num_anchors,
            kernel_size=1
        ),



        # stem1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # pro_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=npro * num_anchors,
            kernel_size=1
        ),
        # alp_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=nalp * num_anchors,
            kernel_size=1
        ),
        # ads0_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads1_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads2_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads3_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads4_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads5_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # cor_pred1
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=8 * num_anchors,
            kernel_size=1
        ),



        # stem2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # pro_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=npro * num_anchors,
            kernel_size=1
        ),
        # alp_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=nalp * num_anchors,
            kernel_size=1
        ),
        # ads0_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads1_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads2_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads3_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads4_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # ads5_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=nads * num_anchors,
            kernel_size=1
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # cor_pred2
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=8 * num_anchors,
            kernel_size=1
        )
    )

    if num_layers == 4:
        head_layers.add_module('stem3',
            # stem3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=1,
                stride=1
            )
        )
        head_layers.add_module('cls_conv3',
            # cls_conv3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('reg_conv3',
            # reg_conv3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('pro_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=npro * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('alp_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=nalp * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('ads0_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=nads * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('ads1_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=nads * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('ads2_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=nads * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('ads3_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=nads * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('ads4_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=nads * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('ads5_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=nads * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('reg_pred3',
            # reg_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=4 * (reg_max + num_anchors),
                kernel_size=1
            )
        )
        head_layers.add_module('cor_pred3',
            # corner_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=8 * num_anchors,
                kernel_size=1
            )
        )

    return head_layers