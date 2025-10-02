import os
import csv
import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MRIDataset(Dataset):
    def __init__(self, split='train', keep_rois=None, n_test_subjects=8):
        self.split = split
        self.keep_rois = keep_rois
        data_csv = f'./indivs_192/combined.csv'

        df = pd.read_csv(data_csv)
        self.scan_paths  = df.iloc[:, 0].tolist()
        self.scores      = df.iloc[:, 2].tolist()   # mental rotation score

        n = len(self.scan_paths)
        # Assume each subject has a single entry in scan_paths
        subject_ids = list(range(n))
        # sort subjects by their score
        sorted_ids = sorted(subject_ids, key=lambda i: self.scores[i])
        # pick evenly spaced indices across the sorted list
        step = max(1, len(sorted_ids) // n_test_subjects)
        test_subjects = set(sorted_ids[::step][:n_test_subjects])

        # build all pairs
        all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

        if split == "test":
            self.pairs = [(i, j) for (i, j) in all_pairs
                          if i in test_subjects or j in test_subjects]
        else:
            self.pairs = [(i, j) for (i, j) in all_pairs
                          if i not in test_subjects and j not in test_subjects]

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        voxel1 = np.load(self.scan_paths[i]).astype(np.float32)
        voxel2 = np.load(self.scan_paths[j]).astype(np.float32)
        label_pth = self.scan_paths[i][:-6] + "labels.npy"
        labels1 = np.load(label_pth).astype(np.int32)
        label_pth = self.scan_paths[j][:-6] + "labels.npy"
        labels2 = np.load(label_pth).astype(np.int32)

        if self.keep_rois is not None:
            mask1 = np.isin(labels1, list(self.keep_rois))
            mask2 = np.isin(labels2, list(self.keep_rois))
            voxel1 = voxel1 * mask1
            voxel2 = voxel2 * mask2

        score1, score2 = float(self.scores[i]), float(self.scores[j])
        target = 0 if score1 > score2 else 1

        # ---- Random flip augmentation only ----
        if self.split == "train":
            if random.random() < 0.5:
                voxel1 = np.flip(voxel1, axis=0).copy()
            if random.random() < 0.5:
                voxel2 = np.flip(voxel2, axis=0).copy()

        scan1 = torch.tensor(voxel1, dtype=torch.float32).unsqueeze(0)
        scan2 = torch.tensor(voxel2, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target, dtype=torch.long)
        return (scan1, scan2), target

    def __len__(self):
        return len(self.pairs)

# ----------------
# ResNet blocks  -
# ----------------
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # the input to the resnet block will be added to the output of that block
        out += self.shortcut(x) 
        return F.relu(out)

# ------------------
# Slicing slicer -
# ------------------
class SlicingNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=8):
        super().__init__()
        # each block has 2 convs - resnet 14
        self.block1 = ResBlock(in_channels, base_channels, stride=2) # (X=H/2, Y=W/2) # will use skip connection bc the input and output of the block are not same res
        self.block2 = ResBlock(base_channels, base_channels) # (X=H/2, Y=W/2)# will not use skip connection bc input and output are the same res
        self.block3 = ResBlock(base_channels, base_channels) # (X=H/2, Y=W/2)
        self.block4 = ResBlock(base_channels, base_channels, stride=2) # (X=H/4, Y=W/4)
        self.block5 = ResBlock(base_channels, base_channels, stride=2) # (X=H/8, Y=W/8)
        #self.block6 = ResBlock(base_channels, base_channels, stride=2) # (X=H/16, Y=W/16)
        #self.block7 = ResBlock(base_channels, base_channels, stride=2) # (X=H/32, Y=W/32)

        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # every slice puts out one neuron 

        # feature dimension output by this slicer
        self.out_dim = base_channels

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        #out = self.block6(out)
        #out = self.block7(out) # shape(B, C=8, X, Y)
        out = self.dropout(out)
        out = self.pool(out) # shape(B, 8, 1, 1)
        return out.flatten(1) # shape(B, 8)

class RankingNet(nn.Module):
    def __init__(self, in_channels, base_channels=16):
        super().__init__()
        self.block1 = ResBlock(in_channels, base_channels, stride=2) # shape(B, C=16, X, Y)
        self.block2 = ResBlock(base_channels, base_channels*2, stride=2) # shape(B, C=32, X, Y)
        self.dropout = nn.Dropout(p=0.3) # shape(B, C=32, X, Y)
        self.pool   = nn.AdaptiveAvgPool2d((1,1)) # shape(B, C=32, X, Y)
        self.fc     = nn.Linear(base_channels*2, 2) # shape(B, C=32, X, Y)

    def forward(self, x): # shape [B, 2, 192, 8, 1]
        out = self.block1(x)
        out = self.block2(out)
        out = self.dropout(out)
        out = self.pool(out)
        return self.fc(out.flatten(1))

class PairwiseBrainNet(nn.Module):
    def __init__(self, slicing_net: nn.Module, ranker: nn.Module, num_slices=160):
        super().__init__()
        self.slicer = slicing_net
        self.ranker  = ranker
        self.num_slices = num_slices
        self.slice_feat_dim = slicing_net.out_dim

    def encode_scan(self, scan):
        B, _, D, H, W = scan.shape
        # (B, D, 1, H, W) # e.g. [2,1,192,192,192]
        # slicing using the batch dim.
        # [2, 192, 1, 192, 192] -> [384, 1, 192, 192]
        scan_2d = scan.permute(0, 2, 1, 3, 4).reshape(B*D, 1, H, W)
        feats = self.slicer(scan_2d) # eg [B (remember we just called B=B*D), 8]
        # now we pull the slices back out into the Depth dimension (i.e. D)
        # print(f'{feats.shape=}') # e.g. [384, 8]
        feats = feats.view(B, D, -1).transpose(1, 2) # [(original!) B, (original!) D, 8]
        # print(f'{feats.shape=}') # e.g. [384, 8] -> [2, 192, 8] -> [2, 8, 192]
        return feats # [B, X=8, D=192]

    def forward(self, scan1, scan2):
        f1 = self.encode_scan(scan1)
        f2 = self.encode_scan(scan2)

        x = torch.cat([f1, f2], dim=1).unsqueeze(-1)
        out = self.ranker(x)

        # enforcing symmetry via hard loss problem
        x_swap = torch.cat([f2, f1], dim=1).unsqueeze(-1)
        out_swap = self.ranker(x_swap) 
        # realistic but likely wrong # [0.4, 0.6] + [0.9, 0.1] = [0.65, 0.35]
        # this is a problem! e.g. ideal soln: [1,0] + [1,0] = [1,0]
        out = (out + out_swap) / 2.0 

        return out

def train(train_dataset, test_dataset, num_epochs=15, grad_accum_steps=1):
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=4, pin_memory=False, persistent_workers=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=2, shuffle=False,
        num_workers=2, pin_memory=False, persistent_workers=False
    )

    # Model
    slicing_net = SlicingNet(in_channels=1, base_channels=8)
    ranking_net = RankingNet(in_channels=slicing_net.out_dim*2, base_channels=16)
    model = PairwiseBrainNet(slicing_net, ranking_net).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f" Epoch {epoch+1}/{num_epochs}")
        train_epoch(model, train_loader, criterion, optimizer, grad_accum_steps)
        test_acc = test_epoch(model, test_loader, criterion)

        if test_acc > best_acc:
            best_acc = test_acc
            fname = f"tmp.pth"
            torch.save(model.state_dict(), fname)

    # reload best weights into the same model object
    model.load_state_dict(torch.load("tmp.pth", map_location=device))
    return model, test_loader

def train_epoch(model, loader, criterion, optimizer, grad_accum_steps=1):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False)
    data_t, step_t = 0.0, 0.0
    end = time.perf_counter()
    for step, ((scan1, scan2), target) in progress:
        data_t += time.perf_counter() - end
        scan1, scan2, target = scan1.to(device), scan2.to(device), target.to(device)
        t0 = time.perf_counter()
        outputs = model(scan1, scan2)
        loss = criterion(outputs, target) / grad_accum_steps
        loss.backward()
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        step_t += time.perf_counter() - t0
        total_loss += loss.item() * grad_accum_steps
        preds = outputs.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)
        avg_loss = total_loss / (step + 1)
        avg_acc = 100.0 * correct / total
        progress.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{avg_acc:.2f}%",
            data_ms=f"{1000.0 * data_t / (step + 1):.1f}",
            step_ms=f"{1000.0 * step_t / (step + 1):.1f}",
        )
        end = time.perf_counter()
    if (step + 1) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    final_acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Train Loss: {total_loss / len(loader):.4f}, Train Accuracy: {final_acc:.2f}%")
    return total_loss / len(loader)

def test_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    progress = tqdm(enumerate(loader), total=len(loader), desc="Testing", leave=False)
    data_t, step_t = 0.0, 0.0
    end = time.perf_counter()
    with torch.no_grad():
        for step, ((scan1, scan2), target) in progress:
            data_t += time.perf_counter() - end
            scan1, scan2, target = scan1.to(device), scan2.to(device), target.to(device)
            t0 = time.perf_counter()
            outputs = model(scan1, scan2)
            loss = criterion(outputs, target)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            step_t += time.perf_counter() - t0
            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            avg_acc = 100.0 * correct / total
            progress.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_acc:.2f}%",
                data_ms=f"{1000.0 * data_t / (step + 1):.1f}",
                step_ms=f"{1000.0 * step_t / (step + 1):.1f}",
            )
            end = time.perf_counter()
    final_loss = total_loss / len(loader)
    final_acc = 100.0 * correct / total
    print(f"\nTest Loss: {final_loss:.4f}, Test Accuracy: {final_acc:.2f}%")
    return final_acc



def rank_subjects(models, dataset):
    """
    Compute ranking for subjects in the given dataset (e.g., test split),
    averaging predictions from all models provided.
    """
    n = len(dataset.scan_paths)
    subject_scores = np.zeros(n)
    subject_counts = np.zeros(n)

    with torch.no_grad():
        for i in range(n):
            for j in range(i+1, n):
                scan1 = torch.tensor(np.load(dataset.scan_paths[i])).unsqueeze(0).unsqueeze(0).float().to(device)
                scan2 = torch.tensor(np.load(dataset.scan_paths[j])).unsqueeze(0).unsqueeze(0).float().to(device)

                probs_accum = torch.zeros(2, device=device)
                for model in models:
                    model.eval()
                    outputs = model(scan1, scan2)
                    probs = torch.softmax(outputs, dim=1)[0]
                    probs_accum += probs

                probs = probs_accum / len(models)  # average over models

                subject_scores[i] += probs[0].item()
                subject_counts[i] += 1
                subject_scores[j] += probs[1].item()
                subject_counts[j] += 1

    avg_scores = subject_scores / subject_counts
    ranked_subjects = sorted(
        [(idx, avg_scores[idx]) for idx in range(n)],
        key=lambda x: x[1],
        reverse=True
    )
    return ranked_subjects


# -------------------------------
# Backward elimination (unchanged)
# -------------------------------
if __name__ == "__main__":
    num_rounds = 4
    num_rois = 60
    keep_rois = set(range(num_rois))

    results = []
    all_models = []

    for round_idx in range(num_rounds):
        roi_accs = []

        # -------------------------------
        # Train this rounds models
        # -------------------------------
        trial_models = []
        n_copies = 1
        for trial in range(n_copies):
            test_dataset = MRIDataset("test",  keep_rois=keep_rois)
            model, test_loader = train(
                MRIDataset("train", keep_rois=keep_rois),
                test_dataset,
                num_epochs=30,
                grad_accum_steps=16
            )
            trial_models.append(model)
        all_models.extend(trial_models)  # <--- add to master list

        # -------------------------------
        # Test each ROI dropout
        # -------------------------------
        for roi in keep_rois:
            trial_accs = []
            for model in trial_models:
                test_dataset = MRIDataset("test",  keep_rois=keep_rois - {roi})
                test_loader = DataLoader(
                    test_dataset, batch_size=4, shuffle=False,
                    num_workers=4, pin_memory=False, persistent_workers=False
                )
                acc = test_epoch(
                    model,
                    test_loader,
                    nn.CrossEntropyLoss().to(device)
                )
                trial_accs.append(acc)

            mean_acc = np.mean(trial_accs)
            roi_accs.append((roi, mean_acc)) 

            results.append({               
                "round": round_idx + 1,
                "roi_test_removed": roi,
                "mean_acc": mean_acc
            })

        # -------------------------------
        # Backward elimination step
        # -------------------------------
        roi_accs.sort(key=lambda x: x[1])
        num_drop = max(1, len(roi_accs) // 3)
        drop_list = [roi for roi, _ in roi_accs[:num_drop]]
        keep_rois -= set(drop_list)

        print(f"\nRound {round_idx+1}: dropped {num_drop} ROIs {drop_list}")
        print(f"Remaining: {len(keep_rois)} ROIs\n")

    # Save results
    out_csv = "roi_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "roi_test_removed", "mean_acc"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {out_csv}")

        # -------------------------------
    # Final test set ranking
    # -------------------------------
    final_test_dataset = MRIDataset("test", keep_rois=keep_rois)
    ranking = rank_subjects(all_models, final_test_dataset)

    print("\nSubject ranking on TEST set (best → worst, averaged over all rounds):")
    for subj_id, score in ranking:
        scan_name = final_test_dataset.scan_paths[subj_id]
        print(f"Subject {subj_id} ({scan_name}): {score:.4f}")


