import torch

def SSVD_trp(protos, label, type_label, contrastive_loss, variance, alpha=0):
    margin = 1.0
    batch_size = protos.size(0)
    extra_term = torch.tensor(0.0).to(protos.device)

    for ids in range(batch_size):
        current_proto = protos[ids]
        current_label = label[ids]
        current_type_label = type_label[ids]
        current_variance = variance[ids]
        positive_ids = (label == current_label)
        negative_ids = ~positive_ids
        
        label_positive_ids = positive_ids * (type_label == 1)
        unlabel_positive_ids = positive_ids * (type_label == 0)
        label_negative_ids = negative_ids * (type_label == 1)
        unlabel_negative_ids = negative_ids * (type_label == 0)
        
        label_positive_distance = torch.tensor(0.0, dtype=torch.float32, device=protos.device)
        if label_positive_ids.any():
            label_positive_distance = torch.mean(torch.square(current_proto - protos[label_positive_ids]), dim=1).sum()
            label_positive_distance = label_positive_distance / len(label_positive_ids)
            if current_type_label == 0:
                var = current_variance
                unlabel_weight = -torch.log(var + 1e-10) * 0.1
                label_positive_distance = unlabel_weight * label_positive_distance
            
        label_negative_distance = torch.tensor(0.0, dtype=torch.float32, device=protos.device)
        if label_negative_ids.any():
            label_negative_distance = torch.mean(torch.square(current_proto - protos[label_negative_ids]), dim=1)
            label_negative_distance = torch.square(torch.clamp(margin - torch.sqrt(label_negative_distance), min=0)).sum()
            label_negative_distance = label_negative_distance / len(label_negative_ids)
            if current_type_label == 0:
                var = current_variance
                unlabel_weight = -torch.log(var + 1e-10) * 0.1
                label_negative_distance = unlabel_weight * label_negative_distance

        unlabel_positive_distance = torch.tensor(0.0, dtype=torch.float32, device=protos.device)
        if unlabel_positive_ids.any():
            unlabel_positive_distance = torch.mean(torch.square(current_proto - protos[unlabel_positive_ids]), dim=1).sum()
            unlabel_positive_distance = unlabel_positive_distance / len(unlabel_positive_ids)
            var = (torch.mean(variance[unlabel_positive_ids]) + current_variance) / 2
            unlabel_weight = -torch.log(var + 1e-10) * 0.1
            unlabel_positive_distance = unlabel_weight * unlabel_positive_distance
        
        unlabel_negative_distance = torch.tensor(0.0, dtype=torch.float32, device=protos.device)
        if unlabel_negative_ids.any():
            unlabel_negative_distance = torch.mean(torch.square(current_proto - protos[unlabel_negative_ids]), dim=1)
            unlabel_negative_distance = torch.square(torch.clamp(margin - torch.sqrt(unlabel_negative_distance), min=0)).sum()
            unlabel_negative_distance = unlabel_negative_distance / len(unlabel_negative_ids)
            var = (torch.mean(variance[unlabel_negative_ids]) + current_variance) / 2
            unlabel_weight = -torch.log(var + 1e-10) * 0.1
            unlabel_negative_distance = unlabel_weight * unlabel_negative_distance
        
        if current_type_label == 1:
            extra_term += unlabel_positive_distance
            contrastive_loss += (0.3 * (label_positive_distance + label_negative_distance) + 0.7 * (unlabel_positive_distance + unlabel_negative_distance))
        else:
            extra_term += label_positive_distance
            contrastive_loss += (0.7 * (label_positive_distance + label_negative_distance) + 0.3 * (unlabel_positive_distance + unlabel_negative_distance))
            
    return contrastive_loss + alpha * extra_term