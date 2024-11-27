'''
Implementation of training module
'''

from tqdm import tqdm
import time
import copy
from torch.optim import lr_scheduler
from tools.utils import visualize_segmentation

'''
train for one epoch
'''
def train(model, dataloader, optimizer, loss_fn):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    total_pixels, correct_pixels = 0, 0

    for images, label_images in tqdm(dataloader):
        images = images.to(DEVICE)
        label_images = label_images.to(DEVICE)

        optimizer.zero_grad()
        y_logits = model(images)
        loss = loss_fn(y_logits, label_images)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        # y_preds: (N, H, W)
        y_preds = torch.argmax(y_logits, axis=1) # argmax along channels (N, C, H, W)
        correct_pixels += (label_images == y_preds).sum().item()
        total_pixels += label_images.numel()

    # Calculate average loss and accuracy for the batch
    train_loss /= len(dataloader)
    train_acc = 100 * (correct_pixels / total_pixels)

    return train_loss, train_acc
'''
Evaluate for one epoch
'''
def evaluate(model, dataloader, optimizer, loss_fn):
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    correct_pixels, total_pixels = 0, 0

    for images, label_images, in dataloader:
        images = images.to(DEVICE)
        label_images = label_images.to(DEVICE)

        y_logits = model(images)
        loss = loss_fn(y_logits, label_images)

        val_loss += loss.item()
        y_preds = torch.argmax(y_logits, axis=1)
        correct_pixels += (label_images == y_preds).sum().item()
        total_pixels += label_images.numel()

    # Average loss/acc over the batches
    val_loss /= len(dataloader)
    val_acc = 100 * (correct_pixels / total_pixels)
    return val_loss, val_acc

'''
Train the model for # Epochs
'''
def train_model(model, 
                train_dataloader, 
                val_dataloader,
                optimizer,
                loss_fn,
                scheduler,
                num_epochs=1):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        start = time.time()
        # Feed forward / backprop on train_dataloader
        train_loss, train_acc = train(model, train_dataloader, optimizer, loss_fn)
        # Feed forward on val_dataloader
        val_loss, val_acc = evaluate(model, val_dataloader, optimizer, loss_fn)

        # Storing epoch histories
        loss_acc_dict['train_loss_lst'].append(train_loss)
        loss_acc_dict['train_acc_lst'].append(train_acc)
        loss_acc_dict['val_loss_lst'].append(val_loss)
        loss_acc_dict['val_acc_lst'].append(val_acc)

        # Update model depending on its peformance on validation data
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

        # Scheduler Update
        scheduler.step()
        
        end = time.time()
        time_elapsed = end - start
        print(f"------------ epoch {epoch} ------------")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}%")
        print(f"Validation loss: {val_loss:.4f} | Validation acc: {val_acc:.2f}%")
        print(f"Time taken: {time_elapsed / 60:.0f}min {time_elapsed % 60:.0f}s")

        if epoch % 10 == 0:
            visualize_segmentation(model, val_dataloader, DEVICE, epoch - 1)

    
    model.load_state_dict(best_model_wts)
    return model