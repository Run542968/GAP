import torch
import torchvision.ops.roi_align as roialign

input = torch.randn((3,4,1,20))
print(input)
proposal = torch.tensor([[[0,1,0,10,0],[0,11,0,15.5,0]],[[1,2.4,0,3.5,0],[1,2.3,0,4.5,0]],[[2,1.2,0,1.5,0],[2,1.2,0,1.5,0]]]).float()
print(input.shape, proposal.shape)
B,Q,_ = proposal.shape
box = proposal.reshape(B*Q,-1)
roi = roialign(input,box,output_size=(1,3))
print(roi.reshape(B,Q,4,-1))
print(roi.shape)