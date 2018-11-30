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
from tensorboardX import SummaryWriter
import listener

# import speller_attention1 as speller
import speller

import Levenshtein
import csv
import sys
from torch.utils.data import DataLoader, TensorDataset
import data_loader_final as data_loader
from data_loader_final import ctc_Dataset
import matplotlib.pyplot as plt

run_id = str(int(time.time()))
os.mkdir('./runs/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)
writer = SummaryWriter('runs/%s' % run_id)

#
def final_test(args, listener_model,speller_model,test_loader,gpu,i):

    read_dictionary = np.load('./data/train_label_dict.npy').item()
    epoch_ls = 0
    listener_model.eval()
    speller_model.eval()
    prediction = []
    flag = 'test'
    with open('submission_basemodel_%d.csv' %(i), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Id'] + ['Predicted'])
        for batch_idx,(data,data_lengths) in enumerate(test_loader):
            # pdb.set_trace()
            # if batch_idx == 50:
            #     break
            data = torch.from_numpy(data).float()
            data_lengths = torch.from_numpy(data_lengths)
            # data = data.view(-1,1,40) #bcs test collate returns 2d
            data = torch.unsqueeze(data, dim=1)
            if gpu is True:
                data = data.cuda()
                data_lengths = data_lengths.cuda()
            attention_key, attention_val, attention_mask = listener_model(data, data_lengths)  # comes out at float
            pred_list = speller_model(None,None, attention_key, attention_val, attention_mask, flag,None)  # batch*lenseq*vocab

            
            sentence = []
            pred_list = torch.cat(pred_list).cpu().numpy()
            for pred in pred_list:
                word = read_dictionary[pred]
                sentence.append(word)
            sentence = "".join(sentence)
            writer.writerow([batch_idx,] + [sentence])
            print('{} {}'.format(batch_idx,sentence))
#
def eval(args, listener_model, speller_model, dev_loader, epoch,gpu):
    listener_model.eval()
    speller_model.eval()
    flag = 'eval'
    prob = np.random.randint(0, len(dev_loader), (1,))[0]
    epoch_loss = 0

    for batch_idx, (data, target,data_lengths,target_lengths,target_mask,target_dict) in enumerate(dev_loader):

        data = torch.from_numpy(data).float()  # THIS HAS TO BE FLOAT BASED ON THE NETWORK REQUIREMENT
        data_lengths = torch.from_numpy(data_lengths).int()  # THIS HAS TO BE LONG BASED ON THE NETWORK REQUIREMENT
        target = torch.from_numpy(target).long()
        target_lengths = torch.from_numpy(target_lengths).int()
        target_mask = torch.from_numpy(target_mask).long()

        if gpu is True:
            data = data.cuda()
            data_lengths = data_lengths.cuda()
            target = target.cuda()
            target_lengths = target_lengths.cuda()
            target_mask = target_mask.cuda()

        attention_key, attention_val, attention_mask = listener_model(data, data_lengths)  # comes out at float
        batch_loss,attention_map = speller_model(target, target_mask, attention_key, attention_val, attention_mask, flag,target_dict)  # batch*lenseq*vocab
        
        ####################### SAVE ATTENTION MASK FOR RANDOM SAMPLE ################
        if batch_idx == prob:
            attention_heatmap = attention_map.cpu().detach().numpy()
            # Create a new figure, plot into it, then close it so it never gets displayed
            fig = plt.figure()
            plt.imshow(attention_heatmap, cmap='hot', interpolation='nearest')
            dir = './models/%s' % run_id
            plt.savefig(dir + '/attention_map %d.png'%(epoch))
            plt.close(fig)

        ####################### LEVENSTEIN DIST #####################################
        # for i in range(target_lengths):
        #     word_loss = Levenshtein.distance(target[i], pred[i])
        #     perplexity = torch.exp(word_loss)
        #     total_word_loss+=word_loss
        #     writer.add_scalar('Eval/Word Loss', word_loss, count)
        #     writer.add_scalar('Eval/Perplexity', perplexity, count)
        epoch_loss += batch_loss.item()
        if batch_idx % 100 == 0:
            print('Eval Epoch: {} \tbatch {} \tLoss: {:.6f}'.format(epoch,batch_idx,batch_loss.item()))
            niter = epoch*len(dev_loader)+batch_idx
            writer.add_scalar('Eval/ Batch Loss', batch_loss.item(), niter)

    # print('Eval Epoch: {} \t Total Word Loss: {:.6f} \tPerplexity: {:.6f} '.format(epoch, total_word_loss/count,perplexity))
    epoch_loss = epoch_loss/len(dev_loader)
    print('---------------------------------')
    print('Eval Epoch: {} \tLoss: {:.6f}'.format(epoch,epoch_loss))
    print('---------------------------------')
    writer.add_scalar('Eval/ Epoch Loss', epoch_loss, epoch)

    return epoch_loss

def train(args, listener_model, speller_model, train_loader,optimizer_speller,optimizer_listener, epoch,gpu):
    listener_model.train()
    speller_model.train()
    flag = 'train'
    epoch_loss = 0
    # Calcualting loss inside speller
    # criterion = nn.CrossEntropyLoss()
    # if gpu is True:
    #     criterion = criterion.cuda()

    for batch_idx, (data, target,data_lengths,target_lengths,target_mask,target_dict) in enumerate(train_loader):
        data = torch.from_numpy(data).float() # THIS HAS TO BE FLOAT BASED ON THE NETWORK REQUIREMENT
        data_lengths = torch.from_numpy(data_lengths).int() #THIS HAS TO BE LONG BASED ON THE NETWORK REQUIREMENT
        target = torch.from_numpy(target).long()
        target_lengths = torch.from_numpy(target_lengths).int()
        target_mask = torch.from_numpy(target_mask).long()

        if gpu is True:
            data = data.cuda()
            data_lengths = data_lengths.cuda()
            target = target.cuda()
            target_lengths = target_lengths.cuda()
            target_mask = target_mask.cuda()
        
        optimizer_speller.zero_grad()
        optimizer_listener.zero_grad()
        attention_key, attention_val, attention_mask = listener_model(data,data_lengths) #comes out at float
        batch_loss,attention_map = speller_model(target, target_mask, attention_key, attention_val, attention_mask, flag,target_dict) #batch*lenseq*vocab

        batch_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.20)
        optimizer_speller.step()
        optimizer_listener.step()
        epoch_loss += batch_loss.item()
        #
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} \tbatch {} \tLoss: {:.6f}'.format(epoch,batch_idx,batch_loss.item()))
            niter = epoch*len(train_loader)+batch_idx
            writer.add_scalar('Train/ Batch Loss', batch_loss.item(), niter)

    epoch_loss = epoch_loss/len(train_loader)       
    print('---------------------------------')
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch,epoch_loss))
    print('---------------------------------')
    writer.add_scalar('Train/Epoch Loss', epoch_loss, epoch)

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
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                                            help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                                            help='number of epochs to train (default: 10)')

    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                                            help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
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

    os.mkdir('./models/%s' % run_id)
    os.mkdir('./models/%s/best' % run_id)
    with open('./models/%s/commandline_args.txt' %run_id, 'w') as f:
        f.write('\n'.join(sys.argv[1:]))

    print('Starting data loading')
    # model.apply(init_randn)
    training_set = ctc_Dataset('train', batch_size=args.batch_size)
    params = {'batch_size': args.batch_size, 'num_workers': args.workers, 'shuffle': True,
              'collate_fn': data_loader.collate}  # if use_cuda else {}
    train_loader = data.DataLoader(training_set, **params)

    validation_set = ctc_Dataset('dev', batch_size= args.batch_size)
    params = {'batch_size': args.batch_size, 'num_workers': args.workers, 'shuffle': False,
              'collate_fn': data_loader.collate}
    validation_loader = data.DataLoader(validation_set, **params)

    listener_model = listener.listenerModel(40,256,128,embed_drop=0,lock_dropi=0.0,lock_droph=0,lock_dropo=0.0)
    speller_model = speller.SpellerModel(training_set.vocab_size,256,512,128)
    if gpu is True:
        listener_model = listener_model.cuda()
        speller_model = speller_model.cuda()

    if args.eval is False:
        print('Done data loading, starting training')
        optimizer_speller = optim.SGD(speller_model.parameters(),lr=args.lr, momentum = args.momentum)
        optimizer_listener = optim.SGD(listener_model.parameters(),lr=args.lr, momentum = args.momentum)

        dir = './models/%s' % run_id
        for epoch in range(args.epochs):
            model_name = 'model_best.pth.tar'
            filepath = os.getcwd() + '/models/1543527265/'+model_name
            state = torch.load(filepath)
            speller_model.load_state_dict(state['speller_state_dict'])
            listener_model.load_state_dict(state['listener_state_dict'])
            train(args, listener_model,speller_model, train_loader,optimizer_speller,optimizer_listener, epoch,gpu)
            eval_loss = eval(args, listener_model,speller_model, validation_loader,epoch,gpu)
            # # ## remember best acc and save checkpoint
            is_best = False
            if best_eval is None or best_eval>eval_loss:
               is_best = True
               best_eval = eval_loss
            model_name = 'model_%d.pth.tar' %(epoch)
            save_checkpoint({
               'epoch': epoch + 1,
               'speller_state_dict': speller_model.state_dict(),
               'listener_state_dict': listener_model.state_dict(),
               'best_acc': best_eval,
               }, is_best,model_name,dir)
    else:
        print('Testing started')
        model_name = 'model_best.pth.tar'
        filepath = os.getcwd() + '/models/1543542219/best/'+model_name
        state = torch.load(filepath)
        speller_model.load_state_dict(state['speller_state_dict'])
        listener_model.load_state_dict(state['listener_state_dict'])
        test_set = ctc_Dataset('test',batch_size=1)
        params = {'batch_size': 1,'num_workers': 1, 'shuffle': False,'collate_fn':data_loader.test_collate } # if use_cuda else {}
        test_loader = data.DataLoader(test_set, **params)
        final_test(args,listener_model,speller_model,test_loader,gpu,3)

if __name__ == '__main__':
        main()
