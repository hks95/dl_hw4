from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
from torch.nn import Sequential
import numpy as np
import time
import pdb
import os
import numpy as np
# from tensorboardX import SummaryWriter
import listener
import speller

# import ctc_model_final as ctc_model
import csv
import sys
# import Levenshtein as L
# from ctcdecode import CTCBeamDecoder
from torch.utils.data import DataLoader, TensorDataset
# from warpctc_pytorch import CTCLoss
import data_loader_final as data_loader
from data_loader_final import ctc_Dataset
# import all.phoneme_list as phonemes

# run_id = str(int(time.time()))
# os.mkdir('./runs/%s' % run_id)
# print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)
# writer = SummaryWriter('runs/%s' % run_id)

# class CTCCriterion(CTCLoss):
#     def forward(self, prediction, target):
#         acts = prediction[0]
#         act_lens = prediction[1].int()
#         label_lens = prediction[2].int()
#         labels = (target).view(-1).int()
#         return super(CTCCriterion, self).forward(
#                 acts=acts,
#                 labels=labels.cpu(),
#                 act_lens=act_lens.cpu(),
#                 label_lens=label_lens.cpu()
#         )
#
# def final_test(args, model,test_loader,gpu,i):
#
#     label_map = [' '] + phonemes.PHONEME_MAP
#     decoder = CTCBeamDecoder(labels=label_map, blank_id=0, beam_width=100)
#     epoch_ls = 0
#     model.eval()
#     prediction = []
#     with open('submission_basemodel_%d.csv' %(i), 'w', newline='') as csvfile:
#         fieldnames = ['Id', 'Predicted']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for batch_idx,(data,data_lengths) in enumerate(test_loader):
#
#             data = torch.from_numpy(data).float()
#             data_lengths = torch.from_numpy(data_lengths).int()
#             data = data.view(-1,1,40) #bcs test collate returns 2d
#             if gpu is True:
#                 data = data.cuda()
#                 data_lengths = data_lengths.cuda()
#
#             logits = model(data,data_lengths)
#             logits = torch.transpose(logits, 0, 1)
#             probs = F.softmax(logits, dim=2).data.cpu()
#
#             output, scores, timesteps, out_seq_len = decoder.decode(probs=probs, seq_lens=data_lengths)
#             pred = "".join(label_map[o] for o in output[0, 0, :out_seq_len[0, 0]])
#             writer.writerow({'Id': batch_idx, 'Predicted': pred})
#
# def eval(args, model,dev_loader, epoch,gpu):
#
#     label_map = [' '] + phonemes.PHONEME_MAP
#     decoder = CTCBeamDecoder(labels=label_map, blank_id=0, beam_width=100)
#     epoch_ls = 0
#     model.eval()
#     for batch_idx,(data, target,data_lengths,target_lengths) in enumerate(dev_loader):
#
#         #pdb.set_trace()
#         data = torch.from_numpy(data).float()
#         data_lengths = torch.from_numpy(data_lengths).int()
#
#         if gpu is True:
#             data = data.cuda()
#             data_lengths = data_lengths.cuda()
#
#         target = np.concatenate(target)
#         target = torch.from_numpy(target).int()
#
#         logits = model(data,data_lengths)
#         logits = torch.transpose(logits, 0, 1)
#         probs = F.softmax(logits, dim=2).data.cpu()
#
#         output, scores, timesteps, out_seq_len = decoder.decode(probs=probs, seq_lens=data_lengths)
#
#         pos = 0
#         ls = 0.
#         #pdb.set_trace()
#         for i in range(output.size(0)):
#             #pdb.set_trace()
#             pred = "".join(label_map[o] for o in output[i, 0, :out_seq_len[i, 0]])
#             true = "".join(label_map[l] for l in target[pos:pos + target_lengths[i]])
#             pos += target_lengths[i]
#             ls += L.distance(pred, true)
#             print(" batch {} ls {}".format(batch_idx,ls))
#         #pdb.set_trace()
#         assert pos == target.size(0)
#         epoch_ls += ls / output.size(0)
#         # print(ls/output.size(0))
#     epoch_ls = epoch_ls/len(dev_loader)
#     print('Test Epoch: {} \t \tL dist: {:.6f}'.format(epoch,epoch_ls))
#     niter = epoch*len(dev_loader)+batch_idx
#     writer.add_scalar('Train/L distance', epoch_ls, niter)
#
#     return epoch_ls


def train(args, listener_model, speller_model, train_loader,optimizer, epoch,gpu):
    # listener_model.train()
    epoch_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    if gpu is True:
        criterion = criterion.cuda()

    for batch_idx, (data, target,data_lengths,target_lengths) in enumerate(train_loader):

        data = torch.from_numpy(data).float() # THIS HAS TO BE FLOAT BASED ON THE NETWORK REQUIREMENT
        data_lengths = torch.from_numpy(data_lengths).int() #THIS HAS TO BE LONG BASED ON THE NETWORK REQUIREMENT
        target = torch.from_numpy(target).long()
        target_lengths = torch.from_numpy(target_lengths).int()

        if gpu is True:
            data = data.cuda()
            data_lengths = data_lengths.cuda()
            target = target.cuda()
            target_lengths = target_lengths.cuda()
        
        optimizer.zero_grad()
        attention_key, attention_val, attention_mask = listener_model(data,data_lengths) #comes out at float
        pred = speller_model(target, target_lengths, attention_key, attention_val, attention_mask) #batch*lenseq*vocab

        target_2 = torch.t(target) #batch size first

        # ignore index part
        for i in range(target_2.shape[0]):
            target_2[i,target_lengths[i]:] = -1

        batch_loss = criterion(pred, target_2.flatten())
        batch_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.20)
        optimizer.step()
        epoch_loss += batch_loss.item()
        #
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} \tbatch {} \tLoss: {:.6f}'.format(epoch,batch_idx,batch_loss.item()))
            # niter = epoch*len(train_loader)+batch_idx
            # writer.add_scalar('Train/ctcLoss', batch_loss_norm.item(), niter)

def save_checkpoint(state,is_best,model_name,dir):
    
    filename=dir+'/' + model_name
    torch.save(state, filename)
    if is_best:
        filename=dir +  '/best/model_best.pth.tar'
        torch.save(state, filename)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch ctc speech')
    parser.add_argument('--batch_size', type=int, default=5, metavar='N',
                                            help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=50, metavar='N',
                                            help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                                            help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                                            help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                                            help='SGD momentum (default: 0.5)')
    parser.add_argument('--use_gpu', type=bool, default=False,
                                            help='decides CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                                            help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                                            help='how many batches to wait before logging training status')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--ctx', type=int, default=14000)
    parser.add_argument('--eval', type=bool, default=False)
    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_gpu is True:
            gpu = True
    else:
            gpu = False

    torch.manual_seed(args.seed)
    if gpu is True:
            torch.cuda.manual_seed(args.seed)

    print('gpu {}'.format(gpu))

    best_eval = None

    # os.mkdir('./models/%s' % run_id)
    # os.mkdir('./models/%s/best' % run_id)
    # with open('./models/%s/commandline_args.txt' %run_id, 'w') as f:
    #     f.write('\n'.join(sys.argv[1:]))

    print('Starting data loading')
    # model.apply(init_randn)
    training_set = ctc_Dataset('train', batch_size=args.batch_size)
    params = {'batch_size': args.batch_size, 'num_workers': args.workers, 'shuffle': True,
              'collate_fn': data_loader.collate}  # if use_cuda else {}
    train_loader = data.DataLoader(training_set, **params)

    validation_set = ctc_Dataset('dev', batch_size=args.test_batch_size)
    params = {'batch_size': args.test_batch_size, 'num_workers': args.workers, 'shuffle': False,
              'collate_fn': data_loader.collate}
    validation_loader = data.DataLoader(validation_set, **params)

    print('Done data loading, starting training')

    listener_model = listener.listenerModel(40,256,128,embed_drop=0,lock_dropi=0.0,lock_droph=0,lock_dropo=0.0)
    speller_model = speller.SpellerModel(training_set.vocab_size,256,512,128)
    if gpu is True:
        listener_model = listener_model.cuda()
        speller_model = speller_model.cuda()

    if args.eval is False:

                        
        optimizer = optim.Adam(listener_model.parameters(),lr=args.lr)

        # dir = './models/%s' % run_id
        for epoch in range(1,2):
            train(args, listener_model,speller_model, train_loader,optimizer, epoch,gpu)
            #model_name = 'model_best.pth.tar'
            #filepath = os.getcwd()+'/models/1541143617/best/' + model_name
            #filepath = os.getcwd()+'/models/1541143617/best/' + model_name
            #state = torch.load(filepath)
            #model.load_state_dict(state['state_dict'])
            #print(model)
            # avg_ldistance = eval(args, model, validation_loader,epoch,gpu)
            ## remember best acc and save checkpoint
            # is_best = False
    #         if best_eval is None or best_eval>avg_ldistance:
    #             is_best = True
    #             best_eval = avg_ldistance
    #         model_name = 'model_%d.pth.tar' %(epoch)
    #         save_checkpoint({
    #             'epoch': epoch + 1,
    #             'state_dict': model.state_dict(),
    #             'best_acc': best_eval,
    #             'optimizer' : optimizer.state_dict(),
    #             }, is_best,model_name,dir)
    # else:
    #     print('Testing started')
    #     model_name = '/model_best.pth.tar'
    #     filepath = os.getcwd() + '/models/1541263511/best/'+model_name
    #     state = torch.load(filepath)
    #     model.load_state_dict(state['state_dict'])
    #     print(model)
    #     test_set = ctc_Dataset('test',batch_size=1)
    #     params = {'batch_size': 1,'num_workers': args.workers, 'shuffle': False,'collate_fn':data_loader.test_collate } # if use_cuda else {}
    #     test_loader = data.DataLoader(test_set, **params)
    #     final_test(args,model,test_loader,gpu,1)

if __name__ == '__main__':
        main()
