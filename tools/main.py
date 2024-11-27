from data.CamVid_Dataset import *
from tools.train import *
from model.FCN_8s import FCN_8s
import torch
from torch import nn
from torch.optim import lr_scheduler
import pickle as pkl

loss_acc_dict = {
    'train_loss_lst' : [],
    'train_acc_lst' : [],
    'val_loss_lst' : [],
    'val_acc_lst' : []
}

if __name__ == '__main__':
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    ''' Step 1 : Getting Ready Data'''
    # Dataset path
    train_img_dir = '/kaggle/input/camvid/CamVid/train'
    train_label_dir ='/kaggle/input/camvid/CamVid/train_labels'
    val_img_dir = '/kaggle/input/camvid/CamVid/val'
    val_label_dir = '/kaggle/input/camvid/CamVid/val_labels'
    # Create Dataset
    train_dataset = CamVidDataset(train_img_dir, train_label_dir, True)
    val_dataset = CamVidDataset(val_img_dir, val_label_dir, False)
    # Create DataLoader
    BATCH_SIZE = 4
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    ''' Step 2 : Getting Ready Model '''
    fcn_8s = FCN_8s(num_classes=32).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fcn_8s.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    ''' Step 3 : Run Training '''
    fcn8s_trained = train_model(fcn_8s,
                                train_dataloader,
                                val_dataloader,
                                optimizer,
                                loss_fn,
                                scheduler,
                                num_epochs=60)

    ''' Step 4 : Save Results '''
    # Saving Losses/Accuracys
    with open('./loss_acc_dict.pkl', 'wb') as f:
        pkl.dump(loss_acc_dict, f)
    # Saving Model
    torch.save(fcn8s_trained.state_dict(), './fcn8s.pth')