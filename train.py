import torch
import numpy as np

def trainer(model, optimizer, train_loader,epoch, criterion, dataset, graph):
    nb_loss = 0
    ct_loss = 0
    sum_loss = 0
    iter_num = 0
    x = []
    y = []
    for batch_idx, (indices, feats) in enumerate(train_loader):
        # 在这里对每个批次的数据进行处理
        # print(batch_idx)
        # print("Batch indices:", indices)
        # print("Batch data shape:", feats.shape)
        feats =feats.cuda()
        out_feats = model(feats) #(b,out_dim)
        # nb_loss, ct_loss, loss= criterion(x=feats, y=out_feats, dataset=dataset, graph=graph)
        nb_loss = criterion(x=feats, y=out_feats, dataset=dataset, graph=graph)
        # print(nb_loss)
        # sum_loss += float(loss)
        
        iter_num += 1
        
        optimizer.zero_grad()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
        # loss.backward()
        nb_loss.backward()
        optimizer.step()
        
        if iter_num == 1:
            x.append(feats)
            y.append(out_feats)

        if iter_num % (len(train_loader) // 2) == 0:
            # print('epoch:{} || iter:{} || nb_loss:{} || ct_loss:{} || sum_loss:{}'.format(epoch, iter_num, nb_loss,ct_loss,
            #                                                            sum_loss/iter))
            print('epoch:{} || iter:{} || nb_loss:{}'.format(epoch, iter_num, nb_loss))
            x.append(feats)
            y.append(out_feats)
        
    return sum_loss / iter_num

def val(model, val_loader, criterion, dataset, graph):
    nb_loss = 0
    ct_loss = 0
    sum_loss = 0
    iter_num = 0
    with torch.no_grad():
        for batch_idx, (indices, feats) in enumerate(val_loader):
            feats = feats.cuda()
            out_feats = model(feats)
            # nb_loss, ct_loss, loss= criterion(x=feats, y=out_feats, dataset=dataset, graph=graph)
            nb_loss = criterion(x=feats, y=out_feats, dataset=dataset, graph=graph)

            # sum_loss += float(loss)
            # loss_avg += float(loss)
            iter_num += 1

    # return sum_loss / iter_num
    return nb_loss.mean()