from Utils.AvgMeter import AvgMeter
from tqdm import tqdm
from Utils.mr2mr import *
import torch
def train_epoch(model, train_loader, optimizer, lr_scheduler, step, accuracies_req):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for batch in tqdm_object:
        for key, value in batch.items():
            if key != 'smiles_input':
                batch[key] = value.to(model.device)

        # Assuming 'model.device' is the target device (e.g., 'cuda' for GPU or 'cpu' for CPU)
        loss, graph2d_loss, surface_loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()

        loss_meter.update(loss=loss.item(),
                          graph2d_loss=graph2d_loss.item(),
                          surface_loss=surface_loss.item())
        loss_meter.get_lr(optimizer)

        tqdm_object.set_postfix(
            train_loss=loss_meter.loss_avg,
            graph2d_loss=loss_meter.graph2d_loss_avg,
            surface_loss=loss_meter.surface_loss_avg,
            lr=loss_meter.lr
        )
        #loss_meter.print_epoch_results() 

    return loss_meter
