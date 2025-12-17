import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import copy
import time
import torch
import random
import numpy as np 
import scipy.ndimage as nd
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from collections import deque
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

effective_bs = 256 # batch_size * grad_accum_steps
train_bs = 256 # train_bs_not_full
test_bs = 64
train_bs_full = 3
grad_accum_steps = round(effective_bs / train_bs)
num_rounds = 1
num_rois = 60
patience = 6 # if test_acc hasn't improved in this many epochs stop
#keep_rois = set(range(num_rois)[1:]) # [1:] skips the blank bkgnd class
#keep_rois = set([29, 12, 4, 31, 49, 5, 35, 53, 48, 51, 52, 1])
hide_tqdm = False # disable tqdm logging
if hide_tqdm: 
    tqdm = lambda *a, **k: __import__('tqdm').tqdm(*a, disable=True, **k)

class MRIDataset(Dataset):
    def __init__(self, split='train', keep_rois=None, n_test_subjects=24, resize_dim=192,
                 slicing_net=None, device=None, preencode=True):
        self.split = split
        self.keep_rois = keep_rois
        self.slicing_net = slicing_net.eval().to(device)
        self.preencode = preencode 
        data_csv = './indivs_192/combined.csv'
        task_idx = 2 # mrt

        # Load CSV and filter NaNs
        df = pd.read_csv(data_csv)
        df = df.dropna(subset=[df.columns[task_idx]])
        self.scan_paths = df.iloc[:, 0].tolist()
        self.scores = df.iloc[:, task_idx].tolist()

        # Split subjects into train/test
        n = len(self.scan_paths)
        subject_ids = list(range(n))
        sorted_ids = sorted(subject_ids, key=lambda i: self.scores[i])
        step = max(1, len(sorted_ids) // n_test_subjects)
        self.test_subjects = set(sorted_ids[::step][:n_test_subjects])
        self.train_subjects = set(subject_ids) - self.test_subjects

        # ---------- precompute once ----------
        self.feature_cache = {}
        with torch.no_grad():  ### now caches in both modes
            for path in tqdm(self.scan_paths, desc=f"Caching slice features ({split})"):
                scan = np.load(path).astype(np.float32)
                labels = np.load(path[:-6] + "labels.npy").astype(np.int32)

                scan_t = torch.from_numpy(scan.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                label_t = torch.from_numpy(labels.astype(np.float32)).unsqueeze(0).unsqueeze(0)

                scan_r = F.interpolate(scan_t, (resize_dim,)*3,
                                       mode="trilinear", align_corners=False)[0, 0]
                label_r = F.interpolate(label_t.float(), (resize_dim,)*3,
                                        mode="nearest")[0, 0].to(torch.int32)
                scan_r[label_r == 0] = 0.0

                if self.keep_rois is not None:
                    roi_mask = torch.isin(label_r, torch.tensor(list(self.keep_rois)))
                    scan_r *= roi_mask

                # either encode or store preprocessed scan directly
                if self.preencode:
                    B, D, H, W = 1, *scan_r.shape
                    slices = scan_r.unsqueeze(1).float().to(device)  # [D, 1, H, W]
                    feats = self.slicing_net(slices).cpu().view(1, D, -1).transpose(1, 2)  # [1,C,D]
                    self.feature_cache[path] = feats.squeeze(0).contiguous()
                else:
                    self.feature_cache[path] = scan_r.clone().float()  # cached resized raw scan

        # Build pairs
        all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j] # 2x data aug
        if split == "test":
            self.pairs = [(i, j) for (i, j) in all_pairs
                          if (i in self.test_subjects) or (j in self.test_subjects)]
        else:
            self.pairs = [(i, j) for (i, j) in all_pairs
                          if (i in self.train_subjects) and (j in self.train_subjects)]

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        p1, p2 = self.scan_paths[i], self.scan_paths[j]
        target = 0 if self.scores[i] > self.scores[j] else 1

        if self.preencode:
            f1 = self.feature_cache[p1] # [8, 96]
            f2 = self.feature_cache[p2]
            if self.split == "train":
                if random.random() < 0.5:
                    f1 = torch.flip(f1, dims=[1])  
                if random.random() < 0.5:
                    f2 = torch.flip(f2, dims=[1])
            x = torch.stack([f1, f2], dim=0)  # [2, 8, 24]
            return x, torch.tensor(target, dtype=torch.long)
        else:
            scan1 = self.feature_cache[p1].unsqueeze(0).unsqueeze(0)  # cached preprocessed scan
            scan2 = self.feature_cache[p2].unsqueeze(0).unsqueeze(0) # [1, 1, 96, 96, 96]
            # ---- Random flip augmentation only ----
            if self.split == "train":
                if random.random() < 0.5:
                    scan1 = torch.flip(scan1, dims=[2])
                if random.random() < 0.5:
                    scan2 = torch.flip(scan2, dims=[2])
        
            return (scan1.float(), scan2.float()), torch.tensor(target, dtype=torch.long)           

    def __len__(self):
        return len(self.pairs)

class ResBlock(nn.Module): # resnet-style blocks
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
        out += self.shortcut(x)
        return F.relu(out)

class SlicingNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=8):
        super().__init__()
        self.block1 = ResBlock(in_channels, base_channels, stride=2)
        self.block2 = ResBlock(base_channels, base_channels)
        self.block3 = ResBlock(base_channels, base_channels)
        self.block4 = ResBlock(base_channels, base_channels, stride=2)
        self.block5 = ResBlock(base_channels, base_channels, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 1

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.dropout(out)
        out = self.pool(out)
        return out.flatten(1)

class RankingNet(nn.Module):
    def __init__(self, in_channels, base_channels=16):
        super().__init__()
        self.block1 = ResBlock(in_channels, base_channels, stride=2)
        self.block2 = ResBlock(base_channels, base_channels*2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.pool   = nn.AdaptiveAvgPool2d((1,1))
        self.fc     = nn.Linear(base_channels*2, 2)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.dropout(out)
        out = self.pool(out)
        return self.fc(out.flatten(1))

class PairwiseBrainNet(nn.Module):
    def __init__(self, slicing_net: nn.Module, ranker: nn.Module):
        super().__init__()
        self.slicer = slicing_net
        self.ranker  = ranker
        

    def encode_scan(self, scan):
        scan = scan.squeeze(1)  # remove extra singleton if present
        B, _, D, H, W = scan.shape
        scan_2d = scan.permute(0, 2, 1, 3, 4).reshape(B*D, 1, H, W)
        feats = self.slicer(scan_2d)
        feats = feats.view(B, D, -1).transpose(1, 2)
        return feats

    def forward(self, scan1, scan2):
        f1 = self.encode_scan(scan1)
        f2 = self.encode_scan(scan2)
        x = torch.stack([f1, f2], dim=1)
        out = self.ranker(x)
        return out

def train_epoch(model, loader, criterion, optimizer, device, grad_accum_steps=1, train_full=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    data_t, step_t = 0.0, 0.0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(enumerate(loader), total=len(loader), desc="Training", leave=False)
    end = time.perf_counter()

    for step, (x, target) in progress: # over the batch
        # measure dataloading delay
        data_t += time.perf_counter() - end
        t0 = time.perf_counter()

        if train_full: # both slicing net and ranking net
            scan1, scan2 = x
            scan1, scan2, target = scan1.to(device), scan2.to(device), target.to(device)
            outputs = model(scan1, scan2)
        else:
            x, target = x.to(device), target.to(device)
            outputs = model.ranker(x)

        loss = criterion(outputs, target) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps
        preds = outputs.argmax(dim=1)
        correct += (preds == target).sum().item()
        total += target.size(0)

        # measure step time
        step_t += time.perf_counter() - t0

        # running averages for tqdm
        avg_loss = total_loss / (step + 1)
        avg_acc = 100.0 * correct / total
        avg_data_ms = 1000.0 * data_t / (step + 1)
        avg_step_ms = 1000.0 * step_t / (step + 1)
        progress.set_postfix(
            loss=f"{avg_loss:.4f}",
            acc=f"{avg_acc:.2f}%",
            data_ms=f"{avg_data_ms:.1f}",
            step_ms=f"{avg_step_ms:.1f}",
        )

        end = time.perf_counter()

    # handle any leftover grads
    if (step + 1) % grad_accum_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    final_acc = 100.0 * correct / total if total > 0 else 0.0
    data_ms = (data_t / len(loader)) * 1000
    step_ms = (step_t / len(loader)) * 1000
    print(f"Train Loss: {total_loss / len(loader):.4f}, Train Accuracy: {final_acc:.2f}%, "
          f"data_ms: {data_ms:.1f}, step_ms: {step_ms:.1f}")
    return total_loss / len(loader), final_acc # final acc is NEW


def test_epoch(model, loader, criterion, device, train_full=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    data_t, step_t = 0.0, 0.0
    progress = tqdm(enumerate(loader), total=len(loader), desc="Testing", leave=False)
    end = time.perf_counter()

    with torch.no_grad():
        for step, (x, target) in progress:
            data_t += time.perf_counter() - end
            t0 = time.perf_counter()

            if train_full:
                scan1, scan2 = x
                scan1, scan2, target = scan1.to(device), scan2.to(device), target.to(device)
                outputs = model(scan1, scan2)
            else:
                x, target = x.to(device), target.to(device)
                outputs = model.ranker(x)

            loss = criterion(outputs, target)
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            total_loss += loss.item()

            step_t += time.perf_counter() - t0

            # running averages for tqdm
            avg_loss = total_loss / (step + 1)
            avg_acc = 100.0 * correct / total
            avg_data_ms = 1000.0 * data_t / (step + 1)
            avg_step_ms = 1000.0 * step_t / (step + 1)
            progress.set_postfix(
                loss=f"{avg_loss:.4f}",
                acc=f"{avg_acc:.2f}%",
                data_ms=f"{avg_data_ms:.1f}",
                step_ms=f"{avg_step_ms:.1f}",
            )

            end = time.perf_counter()

    final_loss = total_loss / len(loader)
    final_acc = 100.0 * correct / total
    data_ms = (data_t / len(loader)) * 1000
    step_ms = (step_t / len(loader)) * 1000
    print(f"\nTest Loss: {final_loss:.4f}, Test Accuracy: {final_acc:.2f}%, "
          f"data_ms: {data_ms:.1f}, step_ms: {step_ms:.1f}")
    return final_loss, final_acc # NEW: final_loss was added

def early_stop_train(train_set, test_set, model, optimizer, criterion, device, train_full=False, plot_path=None):
    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    # 🔹 NEW: track learning curves
    train_losses, test_losses, train_accs, test_accs = [], [], [], []

    # Use full accumulation logic only for train_full, else force 1
    if train_full:
        bs = train_bs_full
        accum_steps = grad_accum_steps  # respect effective_bs formula
    else:
        bs = train_bs
        accum_steps = 1  # force minimal accumulation for ROI ranking (fast mode)

    for _epoch in range(1, 999):
        bs = train_bs_full if train_full else train_bs
        train_loss, train_acc = train_epoch(model,
            DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0),
            criterion, optimizer, device, accum_steps, train_full)
        test_loss, acc = test_epoch(model,
                        DataLoader(test_set, batch_size=test_bs if not train_full else bs, 
                                   shuffle=False, num_workers=0), criterion, device, train_full)
        # 🔹 record
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(acc)

        if acc > best_acc + 1e-4:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break
    model.load_state_dict(best_state)

    # 🔹 Plot learning curves
        # 🔹 NEW: plot learning curves only if a save path is provided
    if plot_path is not None and len(train_accs) > 0:
        epochs = range(1, len(train_accs) + 1)
        plt.figure(figsize=(10, 4))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_accs, label="Train Accuracy")
        plt.plot(epochs, test_accs, label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.title("Accuracy over Epochs")

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, test_losses, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss over Epochs")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()
    return best_acc

def pretrain_slicer(keep_rois, round_idx):
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    slicing_net = SlicingNet(in_channels=1, base_channels=2).to(device)
    ranking_net = RankingNet(in_channels=slicing_net.out_dim*2, base_channels=16).to(device)
    model = PairwiseBrainNet(slicing_net, ranking_net).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=7e-5, weight_decay=1e-2)
    print(f"\nPretraining shared slicer for round {round_idx+1} ...")
    # train on almost all the subjects
    train_set = MRIDataset("train", keep_rois=keep_rois, slicing_net=slicing_net,
                           device=device, preencode=False, n_test_subjects=8)
    test_set  = MRIDataset("test",  keep_rois=keep_rois, slicing_net=slicing_net,
                           device=device, preencode=False, n_test_subjects=8)
    _ = early_stop_train(train_set, test_set, model, optimizer,
                         nn.CrossEntropyLoss().to(device), device, train_full=True)
    slicer_path = f"slicer_round_mrt_{round_idx+1}.pth"
    torch.save(slicing_net.state_dict(), slicer_path)
    print(f"Saved shared slicer weights to {slicer_path}")
    del model
    del slicing_net
    del ranking_net
    torch.cuda.empty_cache()
    print("[Cleanup] Freed GPU memory used by pretraining.")
    return slicer_path
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    slicing_net = SlicingNet(in_channels=1, base_channels=8).to(device)
    ranking_net = RankingNet(in_channels=slicing_net.out_dim * 2, base_channels=16).to(device)
    model = PairwiseBrainNet(slicing_net, ranking_net).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=7e-5, weight_decay=1e-2)

    print(f"\n=== Round {round_idx+1}: Train slicer on ROIs 1–60 except 9, then train ranker on full set ===")

    # ------------------------------------------------------------
    # 🔹 1. TRAIN SLICER (AND RANKER) ON LESION DATASET (EXCLUDING ROI 9)
    # ------------------------------------------------------------
    keep_rois = set(range(1, 60)) - {9}
    print(f"🧩 Training slicer and ranker on lesioned dataset ({len(keep_rois)} ROIs, excluding ROI 9)...")

    train_set = MRIDataset("train", keep_rois=keep_rois,
                           slicing_net=slicing_net, device=device,
                           preencode=False, n_test_subjects=8)
    test_set = MRIDataset("test", keep_rois=keep_rois,
                          slicing_net=slicing_net, device=device,
                          preencode=False, n_test_subjects=8)
    #slicer_acc = 0.0
    #print("HACKING!!! RM ME!!!!")
    slicer_acc = early_stop_train(train_set, test_set, model, optimizer,
                                  criterion, device, train_full=True)
    print(f"✅ Finished training slicer (lesioned data) — Test accuracy: {slicer_acc:.2f}%")

    # ------------------------------------------------------------
    # 🔹 2. FREEZE SLICER WEIGHTS
    # ------------------------------------------------------------
    for param in slicing_net.parameters():
        param.requires_grad = False
    print("🧊 SlicingNet weights frozen — will not update during next stage.")

    # reinitialize ranking_net for fresh training
    ranking_net = RankingNet(in_channels=slicing_net.out_dim * 2, base_channels=16).to(device)
    model = PairwiseBrainNet(slicing_net, ranking_net).to(device)
    optimizer = optim.AdamW(ranking_net.parameters(), lr=7e-5, weight_decay=1e-2)

    # ------------------------------------------------------------
    # 🔹 3. TRAIN RANKING NET ON FULL ROI DATASET (SLICER FROZEN)
    # ------------------------------------------------------------
    keep_rois = set(range(1, 60))  # all 1–60
    print("🔧 Training new ranking network on full ROI dataset (slicer frozen)...")

    train_set = MRIDataset("train", keep_rois=keep_rois,
                           slicing_net=slicing_net, device=device,
                           preencode=True, n_test_subjects=8)
    test_set = MRIDataset("test", keep_rois=keep_rois,
                          slicing_net=slicing_net, device=device,
                          preencode=True, n_test_subjects=8)

    #print("CRAZY HACK!!!")
    #ranker_acc=0
    ranker_acc = early_stop_train(train_set, test_set, model, optimizer,
                                  criterion, device, train_full=False)
    print(f"✅ Finished ranking network training — Test accuracy: {ranker_acc:.2f}%")

    # ------------------------------------------------------------
    # 🔹 4. SAVE WEIGHTS (slicer from lesion, ranker from full)
    # ------------------------------------------------------------
    slicer_path = f"slicer_excl9_trained_round{round_idx+1}.pth"
    ranker_path = f"ranker_full_frozenSlicer_round{round_idx+1}.pth"

    torch.save(slicing_net.state_dict(), slicer_path)
    torch.save(ranking_net.state_dict(), ranker_path)

    print(f"💾 Saved slicer weights to {slicer_path}")
    print(f"💾 Saved ranker weights to {ranker_path}")

    # ------------------------------------------------------------
    # 🔹 5. SUMMARY
    # ------------------------------------------------------------
    print("\n📊 ===== SUMMARY =====")
    print("Phase 1: Train slicer (and ranker) on lesioned data (1–60 except 9)")
    print(f"  → Accuracy: {slicer_acc:.2f}%")
    print("Phase 2: Train ranker only on full ROI dataset (slicer frozen)")
    print(f"  → Accuracy: {ranker_acc:.2f}%")
    print("======================\n")

    del model, slicing_net, ranking_net
    torch.cuda.empty_cache()
    print("[Cleanup] Freed GPU memory used by training.")
    return slicer_path, ranker_path


def worker_process(gpu_id, job_q, result_q, keep_rois, print_lock, slicer_path):
    """Worker that trains ROI-held-out models on one GPU until the queue is empty."""
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    while True:
        try:
            roi = job_q.get_nowait()
        except Exception:
            break
        slicing_net = SlicingNet(in_channels=1, base_channels=8)
        if slicer_path and os.path.exists(slicer_path):
            slicing_net.load_state_dict(torch.load(slicer_path, map_location=device))
        ranking_net = RankingNet(in_channels=slicing_net.out_dim*2, base_channels=16)
        model = PairwiseBrainNet(slicing_net, ranking_net).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=7e-5, weight_decay=1e-2)
        train_set = MRIDataset("train", keep_rois=keep_rois - {roi}, slicing_net=slicing_net, device=device)
        test_set  = MRIDataset("test",  keep_rois=keep_rois - {roi}, slicing_net=slicing_net, device=device)
        acc = early_stop_train(train_set, test_set, model, optimizer,
                               nn.CrossEntropyLoss().to(device), device)
        #acc = early_stop_train(
                                #train_set, test_set, model, optimizer,
                                #nn.CrossEntropyLoss().to(device), device,
                                #plot_path=f"learning_curve_roi_{roi:02d}.png"  # 🔹 NEW
                            #)

        result_q.put((roi, acc))
        with print_lock:
            print(f"[GPU {gpu_id}] Finished ROI {roi:02d} — best test acc {acc:.2f}", flush=True)

def run_parallel_rois(num_gpus, keep_rois, slicer_path):
    """Distribute ROI-held-out models across multiple GPUs dynamically."""
    job_q = mp.Queue()
    result_q = mp.Queue()
    print_lock = mp.Lock()
    for roi in sorted(keep_rois):
        job_q.put(roi)
    processes = []
    for gpu_id in range(num_gpus):
        p = mp.Process(target=worker_process,
                       args=(gpu_id, job_q, result_q, keep_rois, print_lock, slicer_path))
        p.start()
        processes.append(p)
    total = len(keep_rois)
    done = 0
    roi_accs = []
    while done < total:
        roi, acc = result_q.get()
        done += 1
        roi_accs.append((roi, acc))
        print(f"Progress: {done}/{total} models complete.", flush=True)
    for p in processes:
        p.join()
    return roi_accs


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    print("\n=== Train Slicer (ROIs 1–59 except 9) → Freeze → Train Ranker (all 1–59) ===")

    slicer_path, ranker_path = pretrain_slicer(keep_rois=set(range(1, 60)), round_idx=0)

    print(f"✅ Completed training pipeline.")
    print(f"Final slicer saved at: {slicer_path}")
    print(f"Final ranker saved at: {ranker_path}")





# ====================================================================================
# 🔹 NEW (AGE-STYLE): Reconstruct MRT scores using probability 0.5 crossing (like age code)
# ====================================================================================
import numpy as np
import torch
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from tqdm import tqdm

@torch.no_grad()
def infer_all_by_entropy_mrt(model, train_ds, test_ds, device,
                             smooth_sigma=1.0, batch_size=32,use_aug8x=False,
                             plot_examples=True):
    """
    Predict continuous scores (e.g., MRT) using entropy-based regression.
    Compares each test subject to reference subjects with known scores.
    """
    model.eval()

    # --- Group reference features by known MRT score ---
    score_to_feats = {}
    for path, score in zip(train_ds.scan_paths, train_ds.scores):
        if path in train_ds.feature_cache:
            score_to_feats.setdefault(score, []).append(train_ds.feature_cache[path]) # score_to_feats = {2.0: single brain scan}
    ref_scores = sorted(score_to_feats.keys())

    train_pred_scores, train_true_scores = [], []
    test_pred_scores, test_true_scores = [], []

    train_scans, test_scans = [], []
    for i in train_ds.train_subjects:
        train_scans.append(train_ds.scan_paths[i])
    for i in test_ds.test_subjects:
        test_scans.append(test_ds.scan_paths[i])
    # 🔹 CHANGED: define transform for flipping
    import torchvision.transforms as transforms  # 🔹 CHANGED
    flip_h = transforms.RandomHorizontalFlip(p=1.0)  # 🔹 CHANGED

    for ti, (path_test, true_score) in enumerate(
    tqdm(zip(test_ds.scan_paths, test_ds.scores),
         desc="Entropy inference", total=len(test_ds.scan_paths))):

        if true_score == -3.63:
            print("SKIPPING DUMMY")
            continue
            
        x_test = test_ds.feature_cache[path_test].unsqueeze(0).to(device)  # [1, C, D]
        probs_per_ref = []
        for s in ref_scores:
            ref_feats = [p for p in score_to_feats[s] if p != path_test]
            batch_probs = []
            for k in range(0, len(ref_feats), batch_size):
                batch = ref_feats[k:k + batch_size]
                x_refs = torch.stack(batch).to(device)  # [B, C, D]
                if use_aug8x:  # 🔹 CHANGED: new augmentation branch
                    aug_probs = []
                    for x_ref in x_refs:
                        aug_pairs = []
                        for t_flip in [False, True]:
                            for r_flip in [False, True]:
                                for swap in [False, True]:
                                    x_t = flip_h(x_test.squeeze(0)) if t_flip else x_test.squeeze(0)
                                    x_r = flip_h(x_ref) if r_flip else x_ref
                                    pair = torch.stack([x_t, x_r], dim=0)
                                    #print("paiiiirrrr", pair.shape)
                                    if swap:
                                        pair = torch.flip(pair, dims=[0])  # swap test/ref
                                    aug_pairs.append((pair, swap))

                        x_aug = torch.stack([p[0] for p in aug_pairs]).to(device)
                        #print("x_auggg", x_aug.shape)
                        #assert False
                        out = model.ranker(x_aug)
                        p_higher = torch.softmax(out, dim=1)[:, 0].cpu().numpy()
                        for p_val, (_, swapped) in zip(p_higher, aug_pairs):
                            aug_probs.append(1 - p_val if swapped else p_val)
                    batch_probs.append(np.mean(aug_probs))
                else:
                    x_test_rep = x_test.repeat(len(batch), 1, 1)
                    x_pair = torch.stack([x_test_rep, x_refs], dim=1)
                    out = model.ranker(x_pair)
                    p_higher = torch.softmax(out, dim=1)[:, 0].cpu().numpy()
                    batch_probs.extend(p_higher.tolist())
            probs_per_ref.append(np.mean(batch_probs))

        probs = np.array(probs_per_ref)
        if smooth_sigma > 0:
            probs = nd.gaussian_filter1d(probs, sigma=smooth_sigma)

        p = np.clip(probs, 1e-6, 1 - 1e-6)
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        pred_score = ref_scores[np.argmax(entropy)]

        if path_test in train_scans:
            train_pred_scores.append(pred_score)
            train_true_scores.append(true_score)
        else:
            test_pred_scores.append(pred_score)
            test_true_scores.append(true_score)

        if plot_examples and ti < 5:
            plt.figure(figsize=(5,3))
            plt.plot(ref_scores, probs, label='P(test higher)')
            plt.plot(ref_scores, entropy, '--', label='Entropy')
            plt.axvline(pred_score, color='r', linestyle='--', label=f'Pred={pred_score:.2f}')
            plt.axvline(true_score, color='g', linestyle=':', label=f'True={true_score:.2f}')
            plt.xlabel("Reference MRT Score")
            plt.ylabel("Probability / Entropy")
            plt.legend()
            plt.tight_layout()
            save_dir = "entropy_plots_2"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"entropy_plot_{ti:03d}.png"), dpi=200)
            plt.close()  

    train_pred_scores = np.array(train_pred_scores)
    train_true_scores = np.array(train_true_scores)
    test_pred_scores = np.array(test_pred_scores)
    test_true_scores = np.array(test_true_scores)

    train_mae = np.mean(np.abs(train_pred_scores - train_true_scores))
    test_mae = np.mean(np.abs(test_pred_scores - test_true_scores))
    print(f"\n✅ Train MAE = {train_mae:.3f} (N={len(train_pred_scores)})")
    print(f"\n✅ Test MAE = {test_mae:.3f} (N={len(test_pred_scores)})")

    plt.figure(figsize=(5,5))
    plt.scatter(train_true_scores, train_pred_scores, alpha=0.7)
    plt.scatter(test_true_scores, test_pred_scores, alpha=0.7)
    # god forgive marj
    plt.plot([min(list(train_true_scores) + list(test_true_scores)), max(list(train_true_scores) + list(test_true_scores))],
             [min(list(train_true_scores) + list(test_true_scores)), max(list(train_true_scores) + list(test_true_scores))], 'r--', label='Ideal')
    plt.xlabel("True MRT Score")
    plt.ylabel("Predicted MRT (Entropy-based)")
    plt.legend(labels=["train", "test"])
    plt.title(f"Entropy Regression | Train MAE={train_mae:.3f} | Test MAE={test_mae:.3f}")
    plt.tight_layout()
    plt.show()
    return

device = torch.device("cuda:0")

# Load trained slicer
slicing_net = SlicingNet(in_channels=1, base_channels=8).to(device)
slicing_net.load_state_dict(torch.load("slicer_excl9_trained_round1.pth", map_location=device))

# Create datasets (with feature caching)
ranking_net = RankingNet(in_channels=slicing_net.out_dim * 2, base_channels=16)
ranking_net.load_state_dict(torch.load("ranker_full_frozenSlicer_round1.pth", map_location=device))
model = PairwiseBrainNet(slicing_net, ranking_net).to(device)
# Define which ROIs to keep
keep_rois = set(range(num_rois)[1:])
train_ds = MRIDataset("train", keep_rois=keep_rois, slicing_net=slicing_net,
                      device=device, preencode=True, n_test_subjects=8)
test_ds  = MRIDataset("test",  keep_rois=keep_rois, slicing_net=slicing_net,
                      device=device, preencode=True, n_test_subjects=8)

infer_all_by_entropy_mrt(model, train_ds, test_ds, device, use_aug8x=True)