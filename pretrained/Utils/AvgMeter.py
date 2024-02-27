import torch
class AvgMeter:
    def __init__(self):
        self.name = "Metric"
        self.loss_avg = 0.0
        self.loss_sum = 0.0
        self.count = 0
        self.lr = 0.0
        self.surface_loss_avg = 0.0
        self.surface_loss_sum = 0.0
        self.graph2d_loss_avg = 0.0
        self.graph2d_loss_sum = 0.0

    def update(self, loss, graph2d_loss, surface_loss):
        self.count += 1

        self.loss_sum += loss
        self.loss_avg = self.loss_sum / self.count

        self.graph2d_loss_sum += graph2d_loss
        self.graph2d_loss_avg = self.graph2d_loss_sum/self.count
 
        self.surface_loss_sum += surface_loss
        self.surface_loss_avg = self.surface_loss_sum/self.count

    def get_lr(self, optimizer):
        self.lr = optimizer.param_groups[0]['lr']
        return self.lr

