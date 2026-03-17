import torch.nn as nn
import torch


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = targets.long()
        size = log_probs.size()
        targets = torch.zeros((size[0], size[1])).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)

        if self.use_gpu:
            targets = targets.to(torch.device('cuda'))
        targets = (1 - self.epsilon) * targets + self.epsilon / size[1]
        loss = (-targets * log_probs).mean(0).sum()
        return loss

class SupConLoss(nn.Module):
    def __init__(self, device):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = 1.0
    def forward(self, text_features, image_features, t_label, i_targets):
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
            i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float().to(self.device)

        logits = torch.div(torch.matmul(text_features, image_features.T),self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()

        return loss

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

class TripletLoss_WRT(nn.Module):

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss
    
class T2TLoss(nn.Module):
    def __init__(self, margin=0):
        super(T2TLoss, self).__init__()
        self.margin = margin

    def forward(self, rgb_text_features, ir_text_features, rgb_targets, ir_targets):
        n = rgb_text_features.shape[0]

        mask_pos = torch.eq(rgb_targets.expand(n, n), ir_targets.expand(n,n).t()).float()
        dist_mat = pdist_torch(rgb_text_features, ir_text_features) 
        dist_pos = dist_mat * mask_pos 

        dist_rgb = pdist_torch(rgb_text_features, rgb_text_features) 
        dist_ir= pdist_torch(ir_text_features, ir_text_features) 
        mask_neg_rgb = torch.eq(rgb_targets.expand(n, n), rgb_targets.expand(n,n).t()).float()
        mask_neg_ir = torch.eq(ir_targets.expand(n, n), ir_targets.expand(n,n).t()).float()
        dist_neg_rgb, dist_neg_ir, dist_neg_cm = [], [], []
        for i in range(0, n):
            dist_neg_rgb.append( (self.margin - dist_rgb[i][mask_neg_rgb[i] == 0]).clamp(min=0.0).mean() )
            dist_neg_ir.append( (self.margin - dist_ir[i][mask_neg_ir [i] == 0]).clamp(min=0.0).mean() )
            dist_neg_cm.append( (self.margin - dist_mat[i][mask_pos [i] == 0]).clamp(min=0.0).mean() )
        dist_neg_rgb = torch.stack(dist_neg_rgb)
        dist_neg_ir = torch.stack(dist_neg_ir)
        dist_neg_cm = torch.stack(dist_neg_cm)
        
        loss = dist_pos.mean() + dist_neg_rgb.mean() +  dist_neg_ir.mean() +  dist_neg_cm.mean()
        return loss
    
class I2T_TripletLoss(nn.Module):
    def __init__(self, margin=0):
        super(I2T_TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, image_features, text_features, image_targets, text_targets):
        m = image_features.shape[0]
        n = text_features.shape[0]
        
        mask = torch.eq(image_targets.unsqueeze(1).expand(m, n), text_targets.unsqueeze(0).expand(m,n))

        dist = pdist_torch(image_features, text_features) 

        dist_ap, dist_an = [], []
        for i in range(m):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
         # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        return loss

















