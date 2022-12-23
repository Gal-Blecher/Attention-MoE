from imports import *

class MoE(nn.Module):
    def __init__(self, experts, router):
        super(MoE, self).__init__()
        self.n_experts = len(experts)
        if self.n_experts == 2:
            self.expert1, self.expert2 =\
                experts[0], experts[1]
        if self.n_experts == 4:
            self.expert1, self.expert2, self.expert3, self.expert4 =\
                experts[0], experts[1], experts[2], experts[3]
        if self.n_experts == 8:
            self.expert1, self.expert2, self.expert3, self.expert4, self.expert5, self.expert6, self.expert7, self.expert8 =\
                experts[0], experts[1], experts[2], experts[3], experts[4], experts[5], experts[6], experts[7]
        if self.n_experts == 16:
            self.expert1, self.expert2, self.expert3, self.expert4, self.expert5, self.expert6, self.expert7, self.expert8, self.expert9, self.expert10, self.expert11, self.expert12, self.expert13, self.expert14, self.expert15, self.expert16 =\
                experts[0], experts[1], experts[2], experts[3], experts[4], experts[5], experts[6], experts[7], experts[8], experts[9], experts[10], experts[11], experts[12], experts[13], experts[14], experts[15]
        self.router = router


    def forward(self, x):
        if self.n_experts == 2:
            z_1, out_1 = self.expert1(x) # out is logits. logits is probabilities
            z_2, out_2 = self.expert2(x)
            z_list = [z_1, z_2]
            experts_out_list = [out_1, out_2]

        if self.n_experts == 4:
            z_1, out_1 = self.expert1(x) # out is logits. logits is probabilities
            z_2, out_2 = self.expert2(x)
            z_3, out_3 = self.expert3(x)
            z_4, out_4 = self.expert4(x)
            z_list = [z_1, z_2, z_3, z_4]
            experts_out_list = [out_1, out_2, out_3, out_4]

        if self.n_experts == 8:
            z_1, out_1 = self.expert1(x) # out is logits. logits is probabilities
            z_2, out_2 = self.expert2(x)
            z_3, out_3 = self.expert3(x)
            z_4, out_4 = self.expert4(x)
            z_5, out_5 = self.expert5(x)
            z_6, out_6 = self.expert6(x)
            z_7, out_7 = self.expert7(x)
            z_8, out_8 = self.expert8(x)
            z_list = [z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8]
            experts_out_list = [out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8]

        if self.n_experts == 16:
            z_1, out_1 = self.expert1(x) # out is logits. logits is probabilities
            z_2, out_2 = self.expert2(x)
            z_3, out_3 = self.expert3(x)
            z_4, out_4 = self.expert4(x)
            z_5, out_5 = self.expert5(x)
            z_6, out_6 = self.expert6(x)
            z_7, out_7 = self.expert7(x)
            z_8, out_8 = self.expert8(x)
            z_9, out_9 = self.expert9(x) # out is logits. logits is probabilities
            z_10, out_10 = self.expert10(x)
            z_11, out_11 = self.expert11(x)
            z_12, out_12 = self.expert12(x)
            z_13, out_13 = self.expert13(x)
            z_14, out_14 = self.expert14(x)
            z_15, out_15 = self.expert15(x)
            z_16, out_16 = self.expert16(x)
            z_list = [z_1, z_2, z_3, z_4, z_5, z_6, z_7, z_8, z_9, z_10, z_11, z_12, z_13, z_14, z_15, z_16]
            experts_out_list = [out_1, out_2, out_3, out_4, out_5, out_6, out_7, out_8, out_9, out_10, out_11, out_12, out_13, out_14, out_15, out_16]

        z = torch.stack(z_list, dim=0)
        att_weights = self.router(z, z, z).permute(1,0,2)
        experts_out_ = torch.stack(experts_out_list, dim=0).permute(1, 2, 0)
        out = torch.bmm(experts_out_, att_weights)

        return out.squeeze(2), att_weights