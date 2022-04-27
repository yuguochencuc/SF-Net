import sys
import os
import argparse
import torch
from data import *
from Backup import *
from Step2_solver import Solver

# fix random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


def main(args, model):
    tr_dataset = TrainDataset(json_dir=args.json_dir,
                              batch_size=args.batch_size)
    cv_dataset = CvDataset(json_dir=args.json_dir,
                           batch_size=args.cv_batch_size)
    tr_loader = Step2_TrainDataLoader(data_set=tr_dataset,
                                      batch_size=1,
                                      num_workers=args.num_workers,
                                      pin_memory=True)
    cv_loader = Step2_CvDataLoader(data_set=cv_dataset,
                                   batch_size=1,
                                   num_workers=args.num_workers,
                                   pin_memory=True)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # load the model, here model1 refers to the pretrained model, and model2 refers to the decoration model
    [model1, model2] = model
    # a = torch.load(args.pretrained_path)
    model1.load_state_dict(torch.load(args.pretrained_model1_path))
    model1.cuda()
    #model2.load_state_dict(torch.load(args.pretrained_model2_path))
    model2.cuda()
    # print(model2)
    # count the parameter number of the network
    print('The number of trainable parameters of the net is:%d' %(numParams(model2)))
    model2.cuda()
    optimizer = torch.optim.Adam([{'params': model1.parameters(), 'lr': 1e-4},
                                  {'params': model2.parameters()}],
                                  lr=args.lr,
                                  weight_decay=args.l2)
    solver = Solver(data, [model1, model2], optimizer, args)
    solver.train()

# if __name__ == '__main__':
#     args = parser.parse_args()
#     model = train_model
#     print(args)
#     main(args, model)



