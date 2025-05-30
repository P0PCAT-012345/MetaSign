# utils


import logging
import torch
import torch.nn.functional as F
import time
from statistics import mean
from tqdm import tqdm
from collections import defaultdict

# def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, output_progress=True):
#     pred_correct, pred_all = 0, 0
#     running_loss = 0.0
#     pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training...", position=1, leave=False) if output_progress else enumerate(dataloader)
#     for i, data in pbar:
#         l_hands, r_hands, bodies, labels = data
#         l_hands = l_hands.to(device)
#         r_hands = r_hands.to(device)
#         bodies = bodies.to(device)
#         labels = labels.to(device, dtype=torch.long)

#         optimizer.zero_grad()

#         outputs = model(l_hands, r_hands, bodies, training=True)
#         loss = criterion(outputs, labels.squeeze(1))
#         loss.backward()
#         optimizer.step()
#         running_loss += loss

#         _, preds = torch.max(F.softmax(outputs, dim=1), 1)

#         pred_correct += torch.sum(preds == labels.view(-1)).item()
#         pred_all += labels.size(0)


#     if scheduler:
#         scheduler.step()


#     return running_loss, (pred_correct / pred_all) #5352.4126


# def evaluate(model, dataloader, device, output_progress=True):
#     pred_correct, pred_all = 0, 0

#     with torch.no_grad():
#         pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating...", position=1, leave=False) if output_progress else enumerate(dataloader)
#         for i, data in pbar:
#             l_hands, r_hands, bodies, labels = data
#             l_hands = l_hands.to(device)
#             r_hands = r_hands.to(device)
#             bodies = bodies.to(device) 
#             labels = labels.to(device, dtype=torch.long) 

#             for j in range(labels.size(0)):
#                 l_hand = l_hands[j].unsqueeze(0)
#                 r_hand = r_hands[j].unsqueeze(0) 
#                 body = bodies[j].unsqueeze(0) 
#                 label = labels[j]

#                 output = model(l_hand, r_hand, body, training=False)
#                 output = output.unsqueeze(0).expand(1, -1, -1)

#                 if int(torch.argmax(torch.nn.functional.softmax(output, dim=2))) == int(label):
#                     pred_correct += 1
#                 pred_all += 1


#     return (pred_correct / pred_all)


def evaluate_top_k(model, dataloader, device, k=5):
    pred_correct, pred_all = 0, 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), leave=True, total=len(dataloader)):
            l_hands, r_hands, bodies, labels = data
            l_hands = l_hands.to(device)
            r_hands = r_hands.to(device)
            bodies = bodies.to(device)
            labels = labels.to(device, dtype=torch.long)

            for j in range(labels.size(0)):
                l_hand = l_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                r_hand = r_hands[j].unsqueeze(0)  # [1, 204, 21, 2]
                body = bodies[j].unsqueeze(0)  # [1, 204, 12, 2]
                label = labels[j]

                output = model(l_hand, r_hand, body, training=False)
                output = output.unsqueeze(0).expand(1, -1, -1)

                # Statistics
                if int(label[0][0]) in torch.topk(output, k).indices.tolist():
                    pred_correct += 1

                pred_all += 1

    return pred_correct, pred_all, (pred_correct / pred_all)


def get_sequence_list(num):
    if num == 0:
        return [0]

    result, i = [1], 2
    while sum(result) != num:
        if sum(result) + i > num:
            for j in range(i - 1, 0, -1):
                if sum(result) + j <= num:
                    result.append(j)
        else:
            result.append(i)
        i += 1

    return sorted(result, reverse=True)


def train_epoch(model, dataloader, optimizer, device, scheduler=None, output_progress=True):
    model.train()
    correct, total = 0, 0
    running_loss = 0.0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Training...", position=1, leave=False) if output_progress else enumerate(dataloader)

    for i, data in pbar:
        l_support, r_support, b_support, support_labels, l_query, r_query, b_query, query_labels = [d.to(device) for d in data]
        optimizer.zero_grad()

        support_embeddings, query_embeddings = model(l_support, r_support, b_support, l_query, r_query, b_query)

        loss, t, c = model.calculate_loss_and_accuracy(support_embeddings, query_embeddings, query_labels)

        correct += c
        total += t

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        torch.cuda.empty_cache()

    if scheduler:
        scheduler.step()

    return running_loss, (correct / total)


@torch.no_grad()
def evaluate(model, dataloader, device, output_progress=True):
    model.eval()
    correct, total = 0, 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating...", position=1, leave=False) if output_progress else enumerate(dataloader)

    for i, data in pbar:
        l_support, r_support, b_support, support_labels, l_query, r_query, b_query, query_labels = [d.to(device) for d in data]

        support_embeddings, query_embeddings = model(l_support, r_support, b_support, l_query, r_query, b_query)
        t, c = model.calculate_accuracy(support_embeddings, query_embeddings, query_labels)
        correct += c
        total += t
        

    return correct / total


# @torch.no_grad()
# def analyze(model, dataloader, device, output_progress=True):
#     model.eval()

#     pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Evaluating...", position=1, leave=False) if output_progress else enumerate(dataloader)
#     total = 0
#     correct_per_words = torch.tensor([0, 0, 0], device=device)
#     correct_per_label = defaultdict(int)
#     total_per_label = defaultdict(int)

#     for i, data in pbar:
#         l_support, r_support, b_support, support_labels, l_query, r_query, b_query, query_labels = [d.to(device) for d in data]
#         K = l_query.shape[0]
        
#         logits = model(l_support, r_support, b_support, l_query, r_query, b_query)
#         c, t = model.calculate_accuracy(logits, query_labels)
#         correct += c
#         total += t

#     accuracy_per_label = defaultdict(int)
#     for (idx, correct), total in zip(correct_per_label.items(), total_per_label.values()):
#         accuracy_per_label[idx.item()] = correct/total

#     return correct_per_words / total, accuracy_per_label