# 1d. Training, the two evaluation protocols and the analysis (notebook §7–12)

## Purpose
Train every model with **one shared training loop**, evaluate it in **two ways**, and understand *why* the numbers look the way they do.

## Why this component

| Piece | What | Why | Rejected alternative |
|---|---|---|---|
| One generic `train_model` | Same optimiser, loss, schedule for all models | Fair comparison | A custom loop per model |
| AdamW + One-Cycle schedule | Learning rate warms up, then decays | Fast, stable convergence for fine-tuning | Constant learning rate: slower, less stable |
| Class-weighted loss + label smoothing | Rare classes count more; targets are "soft" | Imbalance + more honest confidence | Plain cross-entropy |
| Gradient clipping | Limits the size of each update | Prevents rare exploding updates | – |
| Mixed precision (fp16) | Half-precision maths on GPU | ~2× faster, less memory for transformers | fp32 only: slower, may not fit |
| Best-epoch checkpoint on **validation** | Keep the weights of the best validation epoch | Avoid over-fitted last epochs, no test peeking | Using the last epoch |
| **Protocol 1** random split | 80/10/10 stratified | Comparable with the literature | – |
| **Protocol 2** GroupKFold | Speakers never shared between train and test | The real-user scenario | Reporting only Protocol 1 |
| Per-dataset accuracy | Accuracy per corpus | Explains the combined score | Only one overall number |

## Diagram: the two protocols

```
PROTOCOL 1 – stratified random split                PROTOCOL 2 – 5-fold, speakers kept together
all clips (all speakers mixed)                      speakers:  A B C D E F G H I J …
   │ shuffle, keep emotion proportions                 fold 1: test = {A,F}   train = everyone else
   ├── 80 % train   ← augmentation here only           fold 2: test = {B,G}   train = everyone else
   ├── 10 % validation ← pick the best epoch           …                      (no speaker in both sides)
   └── 10 % test    ← used once                        inside each training part: 10 % of the SPEAKERS
test speakers ALSO appear in train (leak!)                  are held out as validation
```

## Line-by-line walkthrough

### `predict_probs` — run a model on a set of clips
```python
@torch.no_grad()
def predict_probs(model, idx, bs=64, amp=False):
```
`@torch.no_grad()` switches off gradient bookkeeping (we are not training): less memory, faster. `idx` = row numbers of the clips to score.

```python
    model.eval().to(DEVICE)
```
`eval()` turns off dropout/masking and uses BatchNorm's running statistics; `.to(DEVICE)` moves the model to the GPU.

```python
    for xb, _ in make_loader(idx, bs, train=False):
        with torch.autocast(DEVICE.type, enabled=amp and DEVICE.type == "cuda"):
            out.append(torch.softmax(model(xb.to(DEVICE)).float(), dim=-1).cpu())
    return torch.cat(out).numpy()
```
Loop over batches **without augmentation** (`train=False`), run the model, convert logits to probabilities with `softmax` (they add up to 1), move back to the CPU, and glue all batches into one array.

### `train_model` — the training loop
```python
    cfg = ARCH[arch]; seed_everything()
    model = cfg["build"]().to(DEVICE)
```
Look up the model description and build it fresh. Re-seeding before each model gives every run the same random start.

```python
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=LABELS[train_idx])     # weights from the TRAIN fold only
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32, device=DEVICE), label_smoothing=0.05)
```
Class weights come from the **training clips only**; the **loss function** (`crit`) is cross-entropy ("how surprised is the model by the right answer") with those weights and **label smoothing 0.05** (the target "100 % angry" becomes "≈ 99 % angry, a little for the rest" so the model does not become over-confident).

```python
    groups = model.param_groups()
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
```
**AdamW** is the optimiser (the algorithm that updates weights); it uses the parameter groups with their different learning rates; `weight_decay=0.01` gently pulls weights toward zero (regularisation).

```python
    loader = make_loader(train_idx, cfg["bs"], train=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], epochs=epochs,
                                                steps_per_epoch=len(loader), pct_start=0.15)
```
The training loader **with augmentation**. **One-Cycle** raises the learning rate from small to the group's maximum during the first 15 % of training (`pct_start`), then lowers it smoothly to nearly zero. Warm-up protects the pre-trained weights from big early jolts; the decay lets the model settle.

```python
    use_amp = cfg["amp"] and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
```
Mixed precision only if the model asked for it and a GPU exists. The `GradScaler` multiplies the loss to avoid tiny fp16 gradients rounding to zero, then undoes it.

```python
    best_acc, best_state, hist = -1, None, []
    for ep in range(epochs):
        model.train(); t0 = time.time(); tl = n = correct = 0
```
Start tracking the best validation accuracy and the history list. Each **epoch** = one pass over the training data; `model.train()` turns dropout/masking **on**.

```python
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.autocast(DEVICE.type, enabled=use_amp):
                logits = model(xb); loss = crit(logits.float(), yb)
```
For each batch: move data to the GPU; **forward pass** (the model's guess) under mixed precision; compute the loss against the true labels (in fp32 for accuracy).

```python
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
```
Clear old gradients; **backpropagation** (`backward`) computes how each weight should change; `unscale_` restores the true gradient size before clipping.

```python
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
```
**Clip** the total gradient length to at most 1.0 (prevents rare huge steps), apply the weight update, update the scaler, and advance the learning-rate schedule by one step.

```python
            tl += loss.item() * len(yb); n += len(yb); correct += (logits.argmax(1) == yb).sum().item()
```
Accumulate the running loss and the number of correct predictions for the epoch summary.

```python
        va = accuracy_score(LABELS[val_idx], predict_probs(model, val_idx, amp=use_amp).argmax(1))
        hist.append(dict(epoch=ep + 1, train_loss=tl / n, train_acc=correct / n, val_acc=va))
```
After each epoch measure accuracy on the **validation** clips and store the history (used for the training-curve figure and for the report).

```python
        if va > best_acc:
            best_acc = va; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)
```
If this epoch is the best so far, **copy the weights** (to CPU, detached) as a snapshot. At the end, load the best snapshot — so the returned model is the best-validation epoch, not simply the last one.

### Helpers
```python
def per_dataset_table(idx, pred):
    d = pd.DataFrame({"dataset": DATASETS[idx], "ok": (np.asarray(pred) == LABELS[idx])})
    t = d.groupby("dataset").ok.agg(["mean", "count"]); t.columns = ["accuracy", "n_clips"]
    return t
```
Mark each prediction right/wrong, group by source dataset and average: **accuracy per corpus**. This is what explains why the combined score is lower than the TESS/SAVEE scores.

```python
    cm = confusion_matrix(y, p, labels=range(N_CLASSES), normalize="true")
```
(in `plot_cm`) A **confusion matrix**: row = true emotion, column = predicted emotion; `normalize="true"` shows percentages per row, so you can read "of all angry clips, how many were called sad?".

## Protocol 1

```python
all_idx = np.arange(len(df))
tr_idx, tmp_idx = train_test_split(all_idx, test_size=0.2, stratify=LABELS, random_state=SEED, shuffle=True)
va_idx, te_idx = train_test_split(tmp_idx, test_size=0.5, stratify=LABELS[tmp_idx], random_state=SEED, shuffle=True)
```
First cut: 80 % train / 20 % rest. Then split the 20 % half-and-half into **validation** and **test** (10 % each). `stratify=LABELS` keeps the emotion proportions the same in every part; `random_state` makes it repeatable.

```python
shared = set(SPEAKERS[te_idx]) & set(SPEAKERS[tr_idx])
print(f"Leakage check: {len(shared)} of {len(set(SPEAKERS[te_idx]))} test speakers also appear in training (expected -> NOT speaker-independent)")
```
Counts the **speakers present in both train and test** — a deliberate, printed proof that Protocol 1 is *not* speaker-independent.

```python
def run_protocol1(arch):
    if arch in RESULTS["p1"]:
        print(f"[{arch}] Protocol 1 already done (test acc {RESULTS['p1'][arch]['acc']*100:.2f}%) - skipping"); return
```
Resume logic: skip models that already finished.

```python
    model, hist = train_model(arch, tr_idx, va_idx, ARCH[arch]["ep1"], tag=" P1")
    amp = ARCH[arch]["amp"]
    tp, vp = predict_probs(model, te_idx, amp=amp), predict_probs(model, va_idx, amp=amp)
    pred = tp.argmax(1)
```
Train on the 80 %, choose the best epoch on validation; then score the **test set once** (`tp`) and the validation set (`vp`, saved for the optional ensemble). `argmax(1)` = the emotion with the highest probability.

```python
    RESULTS["p1"][arch] = dict(acc=float(accuracy_score(LABELS[te_idx], pred)), macro_f1=float(f1_score(LABELS[te_idx], pred, average="macro")),
                               val_acc=float(accuracy_score(LABELS[va_idx], vp.argmax(1))), n_test=len(te_idx), pred=pred.tolist(),
                               test_probs=tp.round(5).tolist(), val_probs=vp.round(5).tolist(), hist=hist.to_dict("records"))
    save_results()
```
Store accuracy, macro-F1, validation accuracy, all predictions, probabilities and the training history, then write the JSON file. Everything needed for the report figures is kept, so figures can be redrawn **without retraining**.

```python
    del model; torch.cuda.empty_cache()
```
Free the GPU memory before the next model.

### Optional ensemble (off by default)
```python
    best = max((c for r in range(2, len(names) + 1) for c in itertools.combinations(names, r)),
               key=lambda c: accuracy_score(LABELS[va_idx], np.mean([np.array(RESULTS["p1"][a]["val_probs"]) for a in c], 0).argmax(1)))
```
Try every combination of two or more transformers, average their probabilities, and pick the combination with the best **validation** accuracy. Off by default because a deployed system should be one small model.

## Protocol 2 — speaker-independent

```python
def run_protocol2(arch):
    res = RESULTS["p2"].setdefault(arch, {"folds": []})
    done = {f["fold"] for f in res["folds"]}
    for k, (trn, tst) in enumerate(GroupKFold(n_splits=N_FOLDS).split(all_idx, LABELS, groups=SPEAKERS)):
```
`GroupKFold` makes 5 folds so that **all clips of a speaker go to the same side**. `groups=SPEAKERS` is the key argument. `done` lists finished folds for resuming.

```python
        assert not (set(SPEAKERS[trn]) & set(SPEAKERS[tst])), "speaker leakage between train and test!"
```
A built-in **proof**: the set of speakers in training and in testing must be disjoint, otherwise the program stops.

```python
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED).split(trn, groups=SPEAKERS[trn]))   # inner val = unseen speakers
        model, hist = train_model(arch, trn[a], trn[b], ARCH[arch]["ep2"], tag=f" P2 fold{k}")
```
Inside the training part, hold out about 10 % of the **speakers** (again by group) as the validation set, so that choosing the best epoch is also speaker-independent. The test fold is not touched during training.

```python
        pred = predict_probs(model, tst, amp=ARCH[arch]["amp"]).argmax(1)
        acc = float(accuracy_score(LABELS[tst], pred))
```
Score the test fold — people the model has **never heard**.

```python
        if acc >= max(f["acc"] for f in res["folds"]):          # keep the best fold's weights for deployment
            torch.save(model.state_dict(), os.path.join(WORK_DIR, f"best_p2_{arch}_{TAG}.pt")); res["best_fold"] = k
```
Save the weights of the best fold as a `.pt` file – **this is the model that later becomes the ONNX file for the app**.

```python
    accs = [f["acc"] for f in res["folds"]]
    res["mean"], res["std"] = float(np.mean(accs)), float(np.std(accs)); save_results()
```
Mean and standard deviation over the 5 folds: "76 % ± 5" tells the reader both the typical accuracy and how much it varies between groups of speakers.

### Choosing which transformer gets Protocol 2 (no test-set peeking)
```python
ssl_ranked = sorted([a for a in HF_IDS if a in RESULTS["p1"]], key=lambda a: -RESULTS["p1"][a]["val_acc"])
P2_CHOSEN = ssl_ranked[:P2_TRANSFORMERS]
```
Rank the transformers by **validation** accuracy from Protocol 1 and run the expensive 5-fold only for the top one(s). This saves GPU hours and keeps the choice free of test-set information.

## Analysis cells (§11)

* **Comparison table and bar chart** put Protocol 1 and Protocol 2 side by side with the gap (`P1 − P2`).
* **Error analysis:** builds the confusion matrix, lists the most frequent confusions (for example *neutral* ↔ *sad*) — they are the emotions that share pitch/energy patterns.
* **Layer-weights plot:** `torch.softmax(state["layer_w"], 0)` shows which transformer layers the best model preferred; weight on lower/middle layers supports the claim that emotion is mostly acoustic/prosodic information.
* **§12 final results:** builds a markdown table and a **ready-to-paste abstract** from the measured numbers, so no number in the summary is typed by hand.

## Things to tell the mentor
* **Two protocols = intellectual honesty.** In the earlier Kaggle run the Wav2Vec2 model scored 86.5 % on the random split and 76.6 % on unseen speakers; the notebook is designed to show *both*.
* **Where leakage was prevented:** augmentation (train only), class weights (train fold only), model selection (validation only), speaker-disjoint folds (asserted).
* **Why CREMA-D limits the score:** 60 % of the clips, crowd-labelled, with ambiguous emotions; the per-dataset table makes this visible.
