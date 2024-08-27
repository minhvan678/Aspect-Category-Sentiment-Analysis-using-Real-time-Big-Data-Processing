import torch
from my_model import ASPECT

from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.distributed

def train_one_epoch(model, optimizer, criterion, dataloader, epoch,device):

  model.train()
  epoch_loss = 0
  n_iter = len(dataloader)

  with tqdm(dataloader, unit="batch") as tepoch:
    for data in tepoch:
      tepoch.set_description(f"Epoch {epoch}")

      input = data.copy()
      label = input['label']
      del input['label']

      input = input.to(device)
      label = label.to(device)
      logits = model(input)

      loss = 0
      for asp_idx in range(len(ASPECT)):
        l = criterion(logits[:,asp_idx,:],label[:,asp_idx,:].squeeze(-1))
        loss += l

      epoch_loss += loss.item()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
      optimizer.step()
      optimizer.zero_grad()

      tepoch.set_postfix(loss=loss.item())

    epoch_loss /= n_iter
  print(f"At EPOCH {epoch}, loss = {epoch_loss}")

  return epoch_loss


def eval_one_epoch(model, criterion, dataloader, epoch,device):

  print("EVAL MODE:")
  model.eval()

  epoch_loss = 0
  precision = 0
  recall = 0
  f1 = 0

  n_iter = len(dataloader)

  with torch.no_grad():
    for data in dataloader:
      input = data.copy()
      label = input['label']
      del input['label']

      input = input.to(device)
      label = label.to(device)
      logits = model(input)

      # CACULATE VAL_LOSS
      loss = 0
      for asp_idx in range(len(ASPECT)):
        l = criterion(logits[:,asp_idx,:],label[:,asp_idx,:].squeeze(-1))
        loss += l

      epoch_loss += loss.item()

      # CACULATE Precision, Recall, F1-score for each Aspect#Polarity
      num_asp = logits.size()[1]
      n_sample = logits.size()[0]

      ground = label.view(n_sample, num_asp).cpu().detach()
      pred = torch.max(logits.cpu().detach(),2)[1]

      p_batch = 0
      r_batch = 0
      f1_batch= 0
      for idx in range(num_asp):
        p_batch += precision_score(ground[:,idx], pred[:,idx], average='macro',zero_division=0.0,labels = [0,1,2,3])
        r_batch += recall_score(ground[:,idx], pred[:,idx], average='macro',zero_division=0.0,labels = [0,1,2,3])
        f1_batch += f1_score(ground[:,idx], pred[:,idx], average='macro',zero_division=0.0,labels = [0,1,2,3])

      precision += p_batch / num_asp
      recall += r_batch / num_asp
      f1 += f1_batch / num_asp

  epoch_loss /= n_iter
  precision /= n_iter
  recall /= n_iter
  f1 /= n_iter

  print(f"AT epoch {epoch}, val_loss = {epoch_loss}")
  print(f"-- Precision = {precision}, Recall = {recall}, F1-score: {f1}")
  print("======================================")

  return f1

def load_model(path):
    check_point = torch.load(path,map_location=torch.device('cpu'))
    return check_point

def save_model(path, model, epoch):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    },path)

def training(model, optimizer, criterion, train_loader, val_loader, NUM_EPOCHS):
  torch.distributed.init_process_group(backend="nccl")

  count = 0
  best_loss = -10000
  for epoch in range(NUM_EPOCHS):
    train_loss = train_one_epoch(model, optimizer, criterion, train_loader, epoch)
    val_loss = eval_one_epoch(model, criterion, val_loader, epoch)

    if val_loss > best_loss:
      best_loss = val_loss
      count = 0
      print(val_loss,best_loss)
    else:
      print("else", val_loss,best_loss)
      count += 1
      if count == 5:
        print("EARLY STOP")
        break

  torch.destroy_process_group()
