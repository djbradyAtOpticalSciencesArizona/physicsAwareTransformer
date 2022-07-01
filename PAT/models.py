import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PAT(nn.Module):
    def __init__(self, upscale_factor, in_channel=3, out_channel=3, num_input=2):
        super(PAT, self).__init__()
        ### feature extraction
        self.init_feature = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResB(64),
            ResASPPB(64),
            ResB(64),
            ResASPPB(64),
            ResB(64),
        )
        ### paralax attention
        self.pam = PAM(64, num_input)
        ### upscaling
        self.upscale = nn.Sequential(
            ResB(64),
            ResB(64),
            ResB(64),
            ResB(64),
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, out_channel, 3, 1, 1, bias=False)
        )
    def forward(self, x_left, x_rights, is_training, Pos):
        if not isinstance(x_rights, list):
            x_rights = [x_rights]
        ### feature extraction
        buffer_left = self.init_feature(x_left)
        buffer_rights = [self.init_feature(x_right) for x_right in x_rights]
        if is_training == 1:
            ### parallax attention
            buffer, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = self.pam(buffer_left, buffer_rights, is_training, Pos)
            ### upscaling
            out = self.upscale(buffer)
            return out, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
                   (V_left_to_right, V_right_to_left)
        if is_training == 0:
            ### parallax attention
            buffer, M_right_to_left = self.pam(buffer_left, buffer_rights, is_training, Pos)
            ### upscaling
            out = self.upscale(buffer)
            return out, M_right_to_left

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class ResASPPB(nn.Module):
    def __init__(self, channels):
        super(ResASPPB, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.b_1 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_2 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_3 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))

        buffer_2 = []
        buffer_2.append(self.conv1_2(buffer_1))
        buffer_2.append(self.conv2_2(buffer_1))
        buffer_2.append(self.conv3_2(buffer_1))
        buffer_2 = self.b_2(torch.cat(buffer_2, 1))

        buffer_3 = []
        buffer_3.append(self.conv1_3(buffer_2))
        buffer_3.append(self.conv2_3(buffer_2))
        buffer_3.append(self.conv3_3(buffer_2))
        buffer_3 = self.b_3(torch.cat(buffer_3, 1))

        return x + buffer_1 + buffer_2 + buffer_3
    
class fePAM(nn.Module):
    def __init__(self):
        super(fePAM, self).__init__()
        self.softmax = nn.Softmax(-1)
    def forward(self, Q, S, R, Pos, is_training):
        ## Q: n_batch x C x h x w
        ## S, R: n_batch x C x H x W
        ## Pos: xxs: nparray, n_batch x h x w x k; yys: nparray, n_batch x h x w x k
        n_batch, n_channel, h, w = Q.size()
        
        xxs, yys = Pos
        Key = []
        Value = []
        for i in range(n_batch):
            Pos_x, Pos_y = xxs[i].flatten().long(), yys[i].flatten().long() #(h*w*k, )
            Key.append(S[i, :, Pos_x, Pos_y]) 
            Value.append(R[i, :, Pos_x, Pos_y])
        Key = torch.stack(Key, dim=0) #n_batch x C x h*w*k
        Value = torch.stack(Value, dim=0)#n_batch x C x h*w*k

        Key = Key.view(n_batch, n_channel, h*w, -1).permute(0, 2, 1, 3) #n_batch x h*w x C x k
        Q = Q.permute(0, 2, 3, 1).view(n_batch, h*w, n_channel).unsqueeze(2) # n_batch x h*w x 1 x C
        score = torch.matmul(Q, Key) #n_batch x h*w x 1 x k
        M_right_to_left = self.softmax(score) #n_batch x h*w x 1 x k
#         ### HARDMAX when inference ### 
#         ### 5/10/22 ##
#         if is_training == 0:
#             k = M_right_to_left.size(-1)
#             M_right_to_left = nn.functional.one_hot(torch.argmax(M_right_to_left, dim=-1), num_classes=k).float()
#         ##############################
        
        Value = Value.view(n_batch, n_channel, h*w, -1).permute(0, 2, 3, 1) #n_batch x h*w x k x C
        buffer = torch.matmul(M_right_to_left, Value) #n_batch x h*w x 1 x C
        buffer = buffer.squeeze().view(n_batch, h, w, n_channel).permute(0, 3, 1, 2) #n_batch x C x h x w

        return buffer, M_right_to_left

class PAM(nn.Module):
    def __init__(self, channels, num_input=2):
        super(PAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2s = nn.ModuleList([nn.Conv2d(channels, channels, 1, 1, 0, bias=True) for i in range(num_input-1)])
        self.b3s = nn.ModuleList([nn.Conv2d(channels, channels, 1, 1, 0, bias=True) for i in range(num_input-1)])
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(64)
        self.fe_pam = fePAM()
        self.fusion = nn.Conv2d(channels * num_input, channels, 1, 1, 0, bias=True)# + 1, channels, 1, 1, 0, bias=True)
    def __call__(self, x_left, x_rights, is_training, Poss):
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        if not isinstance(x_rights, list):
            x_rights = [x_rights]
        if not any(isinstance(i, list) for i in Poss):
            Poss = [Poss]
        buffer_rights = [self.rb(x_right) for x_right in x_rights]

        ### M_{right_to_left}
        Q = self.b1(buffer_left)#.permute(0, 2, 3, 1)                                                # B * H * W * C
        Ss, Rs = [], []
        for i in range(len(buffer_rights)):
            Ss.append(self.b2s[i](buffer_rights[i]))
            Rs.append(self.b3s[i](buffer_rights[i]))
        #Ss = [self.b2(buffer_right) for (self.b2, buffer_right) in zip(self.b2s, buffer_rights)]#.permute(0, 2, 1, 3)  # B * H * C * W
#         score = torch.bmm(Q.contiguous().view(-1, w, c),
#                           S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
#         M_right_to_left = self.softmax(score)

        ### fusion
        #Rs = [self.b3(buffer_right) for (self.b3, buffer_right) in zip(self.b3s, buffer_rights)]
#         buffer = R.permute(0,2,3,1).contiguous().view(-1, w, c)                      # (B*H) * W * C
#         buffer = torch.bmm(M_right_to_left, buffer).contiguous().view(b, h, w, c).permute(0,3,1,2)  #  B * C * H * W
        buffers = []
        M_right_to_lefts = []
        for S, R, Pos in zip(Ss, Rs, Poss):
            buffer, M_right_to_left = self.fe_pam(Q, S, R, Pos, is_training)
            buffers.append(buffer)
            M_right_to_lefts.append(M_right_to_left)
        buffers.append(x_left)
        out = self.fusion(torch.cat(tuple(buffers), 1))#, V_left_to_right), 1))

        ## output
        if is_training == 1:
            return out, (None, None), (None, None), (None, None)#\
               #(M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)), \
               #(M_left_right_left.view(b,h,w,w), M_right_left_right.view(b,h,w,w)), \
               #(V_left_to_right, V_right_to_left)
        if is_training == 0:
            return out, M_right_to_lefts
