from imports import *

class MoE(nn.Module):
    def __init__(self, experts, router):
        super(MoE, self).__init__()
        self.expert1, self.expert2 =\
            experts[0], experts[1]
        self.router = router


    def forward(self, x):
        z_1, out_1 = self.expert1(x) # out is logits. logits is probabilities
        z_2, out_2 = self.expert2(x)
        z_list = [z_1, z_2]
        z = torch.stack(z_list, dim=0)
        att_weights = self.router(z, z, z).permute(1,0,2)
        experts_out_list = [out_1, out_2]
        experts_out_ = torch.stack(experts_out_list, dim=0).permute(1, 2, 0)
        out = torch.bmm(experts_out_, att_weights)

        return out.squeeze(2), att_weights