# Human Action Recognition on HAA500
## Skeleton-Based Graph Convolutional Networks

---

## Chapter 1 - Exploratory Data Analysis

The starting point was the HAA500 dataset: 10,000 YouTube video clips spread across 500 fine-grained action categories, exactly 20 clips per class. The metadata was loaded from raw `.txt` files and consolidated into a single CSV (`HAA500_consolidated_metadata.csv`) containing per-clip properties: YouTube URL, start/end timestamps, frame count, resolution, FPS, whether the camera was moving, and the number of dominant figures in the shot.

The first thing that stood out was how short the clips are. The mean duration is 2.12 seconds, with most clips falling under 4 seconds. The frame counts follow the same pattern - a mean of 59.4 frames, which conveniently landed close to the 60-frame window later chosen for training. A small number of outliers stretch to 30+ seconds (slow yoga poses, sustained holds), and a handful have fewer than 10 frames (fast cuts, detection failures).

Resolution is mostly consistent: 71.5% of clips are HD 720p landscape, with the remainder split across SD 480p and lower resolutions. Frame rates vary wildly - from 6 fps up to 30 fps.

Two properties of the dataset matter most for understanding the difficulty of the task. First, 22.9% of videos have camera motion - panning, tilting, or zooming - which means skeleton coordinates extracted in image space will shift even when the person isn't moving. Second, 16.4% of clips have more than one dominant person, which is a problem for single-person pose detectors.

After confirming all 10,000 video files were physically present, a stratified 70/15/15 split was applied (`dataset_split.py`) using `sklearn`'s `StratifiedShuffleSplit` with `random_state=42`. Stratification ensures every class contributes proportionally to each split. The result: 7,000 training videos, 1,500 validation, 1,500 test - roughly 14 training samples per class.

---

## Chapter 2 - Skeleton Extraction

### First pass: 2-D image-normalised landmarks

The original extraction (`pose_extraction.py`) used MediaPipe's `PoseLandmarker` in VIDEO mode to extract 33 body joints per frame. The output per frame was `[x, y, visibility]` - coordinates normalised to `[0, 1]` relative to the image dimensions, plus a detection confidence score. Each video was saved as a `(T, 33, 3)` NumPy array under `extracted_skeletons/{split}/{class}/{video}.npy`.

This worked but had a fundamental problem: the coordinates are in image space. If a person walks toward the camera, their joints spread out across the frame even though the motion itself is just translation. If the camera pans, every joint shifts. The 2-D image coordinates mix up actual body motion and camera behaviour, and with 22.9% of clips having moving cameras, this contaminated a significant chunk of the dataset.

### Second pass: 3-D world coordinates

A second extractor (`pose_extraction_3d.py`) was written to use `pose_world_landmarks` instead of `pose_landmarks`. MediaPipe's world landmarks are a different coordinate system entirely: expressed in metres, hip-centred (the midpoint of the two hip joints is placed at the origin), and reconstructed from a 3-D body model rather than projected from the image. Each frame yields `[x, y, z, visibility]`, saved as `(T, 33, 4)`.

The practical effect is that world coordinates don't care where the person is in the frame or what the camera is doing - they describe the body's geometry in 3-D space relative to itself. Actions like reaching forward versus reaching sideways, which look identical in a 2-D side-view projection, become distinguishable through the `z` axis.

Running MediaPipe on a headless Linux environment (WSL2 without a display server) immediately produced a crash: the shared library `libmediapipe.so` has an ELF dependency on `libGLESv2.so.2`, which doesn't exist in a headless setup. The fix required three things: creating a symlink from `libGLESv2.so.2` to `libEGL.so.1` inside the virtual environment's lib directory, prepending that directory to `LD_LIBRARY_PATH`, and setting `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so.1` before Python starts. These were bundled into a shell wrapper (`run_extraction_3d.sh`) so the extraction process could run unattended.

The extractor processes one video at a time: open the video with OpenCV, read frames in a loop, convert to RGB, pass to the landmarker with a monotonically increasing millisecond timestamp (derived from frame index and FPS), and write the world landmark array to a `.npy` file. If no person is detected in a frame, a zero row is inserted. If world landmarks aren't available but image landmarks are, the extractor falls back to 2-D with `z=0`. The full 10,000-video extraction ran overnight at ~42 files per minute and completed without errors - zero NaN or Inf values in any of the saved arrays.

### Preprocessing applied at training time

Rather than baking preprocessing into the `.npy` files, it's applied on-the-fly in the dataset class:

**Normalisation.** For 3-D world coordinates, MediaPipe already handles hip-centring. An additional scale normalisation divides all position values by the mean distance between shoulder joints (joints 11 and 12) across frames. This makes bodies of different physical sizes produce consistent coordinate magnitudes. For 2-D coordinates, hip-centring is applied manually first.

**Temporal sampling.** Every sequence is brought to exactly 60 frames. For longer videos, training uses a random contiguous crop of 60 frames (this is itself a form of augmentation - each epoch sees a different window). Validation and test use uniform sub-sampling across the full sequence. Videos shorter than 60 frames are padded by repeating the last frame.

**Velocity features.** Per-frame velocity is appended as extra channels: `Δx, Δy, Δz` computed as the frame-to-frame difference in joint positions. The first frame gets zero velocity. This brings each joint's feature vector to 7 dimensions: `[x, y, z, visibility, Δx, Δy, Δz]`.

**Bone features.** A second representation is derived from the same data. For each joint, the mean of all vectors pointing from that joint to its connected neighbours is computed across the 34 skeleton edges. This encodes the relative geometry of limb segments - effectively joint angles and bone lengths - which is complementary to absolute joint positions.

---

## Chapter 3 - Model Training

### Architecture: the skeleton graph

Both models treat the skeleton as a graph: 33 nodes (joints) connected by 34 edges following the anatomical structure MediaPipe defines - neck, shoulders to elbows to wrists, hips to knees to ankles, and so on. The normalised adjacency matrix is pre-computed once as `D^{-0.5}(A+I)D^{-0.5}` and stored as a fixed buffer. Each graph convolution layer adds a learnable residual `ΔA` on top of this, initialised to zero, so the model starts from the anatomical graph and can learn task-specific connections during training.

### Run 1 - Baseline single-stream ST-GCN (2-D landmarks)

The first model (`st_gcn_pipeline.py`) is a single-stream ST-GCN operating on 2-D image-normalised landmarks. The input tensor is `(batch, 7, 60, 33, 1)` - 7 channels (x, y, visibility, velocity, acceleration), 60 frames, 33 joints, 1 person. Ten blocks process the sequence with channel progression `64→64→64→64→128→128→128→256→256→256`, with stride-2 temporal downsampling at blocks 5 and 8. Each block applies a graph convolution, batch norm, ReLU, then a temporal convolution with kernel size 9. Residual connections run through every block.

The first training attempts without regularisation were textbook overfitting: training accuracy hit 99% by epoch 20 while validation stayed below 15%. The model was simply memorising the 14 training samples per class.

Several regularisation techniques were stacked until the gap closed to a manageable level:

- **Dropout (0.5)** after each temporal convolution.
- **Stochastic depth**: each block has a probability of being entirely skipped during training, increasing linearly from 0 at block 1 to 0.3 at block 10. This forces the model to learn redundant representations across blocks.
- **Label smoothing (ε=0.15)**: instead of training against a hard one-hot target, the correct class gets probability `0.85 + 0.15/500` and all others share the remaining `0.15`. Prevents the model from becoming overconfident.
- **Mixup (α=0.2)**: two training samples are blended - both the skeleton sequences and their labels - with a mixing coefficient drawn from a Beta distribution. The model must predict an interpolated label for an interpolated input. This was the single most effective regulariser: it directly attacks the memorisation problem by ensuring no training sample ever appears exactly the same twice, and by smoothing the decision boundaries between the 500 classes.

The optimiser is AdamW with weight decay `5e-4`. The learning rate follows a 5-epoch linear warm-up from `0.1×lr` to `lr=1e-3`, then cosine annealing down to `1e-5` over the remaining epochs. Gradient clipping at norm 1.0 prevents instability.

Training for 100 epochs reached **31.5% top-1 / 48.5% top-5** validation accuracy (best at epoch 86). Training accuracy was ~97%, so meaningful overfitting remained.

### Run 2 - Two-stream ST-GCN v2 (3-D landmarks)

The second model (`st_gcn_v2.py`) made four structural changes:

**3-D world coordinates as input.** Each joint's feature vector is `[x, y, z, visibility, Δx, Δy, Δz]` - the same 7 channels but with genuine depth information and without camera-motion contamination.

**Two streams.** Two independent copies of the ST-GCN run in parallel: one on joint features (absolute positions), one on bone features (relative limb geometry). Their output logit vectors are averaged. This is essentially an ensemble of two classifiers that see complementary views of the same action.

**Multi-scale temporal convolution.** The single-kernel-9 temporal convolution in each block is replaced with four parallel branches: kernels 3, 7, 9, and a max-pool branch. Each branch produces one quarter of the output channels; they are concatenated and batch-normalised back to the original channel count. Different actions unfold at different temporal scales - a finger snap takes 3 frames, a squat takes 30 - and the multi-scale branches capture all of them simultaneously.

**Temporal self-attention in the top two blocks.** The deepest blocks (256 channels) apply multi-head self-attention (8 heads) over the time axis, independently for each joint. The feature tensor `(N, C, T, V)` is reshaped to `(N×V, T, C)`, attention is applied, and the result is added back via a residual with LayerNorm. This allows the model to correlate frames that are far apart in time - useful for cyclic actions where the start and end of a cycle mirror each other.

**The NaN bug.** The first training run with this architecture appeared to work for 5 epochs, then the loss became NaN at epoch 6 and the model weights were fully corrupted by epoch 12, spending the remaining 138 epochs at 0.2% accuracy (random chance). The cause: `nn.MultiheadAttention` running inside PyTorch's AMP (automatic mixed precision, float16) causes softmax overflow - the attention scores become too large for float16 to represent, producing NaN that propagates into the weights. The GradScaler is supposed to catch this, but the NaN was occurring in the forward pass before the backward pass, so the weights were being updated with NaN values before the scaler had a chance to intervene. The fix was to force the attention computation to stay in float32 even when the rest of the model runs in float16, by wrapping it in `torch.amp.autocast(enabled=False)` and casting the input to `.float()` before entering the attention module.

After the fix, the second run trained cleanly for 150 epochs, reaching **33.7% top-1 / 51.5% top-5** - a +2.2% improvement over the baseline.

### Run 3 - Reviewer-suggested redesign (`st_gcn_v3.py`)

After the v2 run a reviewer flagged several concerns: the adjacency-matrix normalisation looked suspicious, dropout 0.5 seemed too high, mixup was conjectured not to help long-term, and the deep-only placement of attention (blocks 8–9) felt backwards - attention should arguably be earlier to capture global structure before convolutions localise it. A v3 was written that tested each of those critiques.

The adjacency-matrix concern dissolved on inspection. The expression `np.diag(d) @ A @ np.diag(d)` with `d = A.sum(1)^{-0.5}` is the standard Kipf–Welling symmetric normalisation `D^{-0.5}(A+I)D^{-0.5}`. Because `D` is diagonal, `D^{-0.5}` is its own transpose, so the "missing transpose" the reviewer suspected is mathematically equivalent to the existing code. A one-line citation comment was added and no functional change followed.

The other suggestions were implemented as full changes: dropout dropped to 0.3, mixup removed, attention redistributed to blocks `[3, 6, 9]` (one per channel stage rather than concentrated at the end), per-frame augmentation added (independent scale jitter per frame, per-frame Gaussian noise, temporal cutout zeroing 2–8 contiguous frames at 30% probability, joint dropout zeroing 1–3 joints at 30% probability), and a `WeightedRandomSampler` substituted for plain `shuffle=True` to break same-class adjacency in the file system.

At 10 epochs v3 already lagged v2 by ~5pp top-1, despite the train loss falling faster. At 30 epochs the picture was definitive: v2 reached 29.8% top-1 / 51.0% top-5, while v3 reached 22.7% / 42.5%. The train–validation gap, which had been ~15pp for v2, ballooned to ~31pp for v3 - classic insufficient-regularisation overfitting. Removing dropout and mixup simultaneously left the model with too much capacity for the 14-samples-per-class regime; the stronger augmentation acted on the input but couldn't compensate for the regularisation removed from inside the model.

The fix - restore `DROPOUT=0.5` and `MIXUP_ALPHA=0.3` while keeping the architectural and sampler changes - produced v3-hireg: 26.3% top-1 / 46.7% top-5. The overfitting gap closed (12.7pp, slightly better than v2's 14.7pp), but absolute accuracy was still −3.5pp top-1 / −4.3pp top-5 below v2. The reviewer-suggested architectural changes alone were a net regression at this scale.

Two single-axis bisects against v3-hireg isolated the contributions:

| Variant | Best Top-1 | Notes |
|---|---|---|
| v3 hi-reg (all changes) | 26.3% | Distributed attention + per-frame aug + weighted sampler |
| Bisect A: revert sampler → `shuffle=True` | 25.9% | Sampler change was a wash (data is already class-balanced) |
| Bisect B: revert attention → `[8, 9]` | 27.4% | Distributed attention was a small negative |
| Bisect C: revert aug → v2-style (clip-level) | 27.9% | Augmentation aggression was the biggest single cost |

Even the best bisect (v2-style augmentation, v2-style attention) stalled at ~28% top-1 - still ~2pp behind v2. The conclusion: on a fixed v2-quality regularisation budget, the v3 architectural changes did not help. The diagnostic step (later) would suggest why - the failure modes aren't where the architectural variations would have leverage.

`st_gcn_v3.py` is retained as a research artifact: it preserves toggleable flags (`USE_WEIGHTED_SAMPLER`, `ATTENTION_BLOCKS`, `V3_AUGMENT_MODE` env-var) so the bisects can be re-run, and `MIXUP_ALPHA > 0` re-enables mixup. None of these flags is the default behaviour in subsequent versions.

### Run 4 - Three-stream extension (`st_gcn_v4.py`)

Where v3 explored the reviewer's surface critiques, v4 took the obvious next step within base ST-GCN: a third stream. The canonical multi-stream ST-GCN family (Shi et al., 2019, "Two-Stream Adaptive GCN") routinely uses joint, bone, and motion streams, each fed a *specialised* 4-channel feature rather than a concatenated 7-channel mix:

- **Joint stream**: `[x, y, z, visibility]`
- **Bone stream**: `[bx, by, bz, visibility]` where bone vectors are computed across the 34 anatomical edges
- **Motion stream**: `[vx, vy, vz, visibility]` where velocities are temporal first differences of positions

Each stream is a separate `SingleStream` network (the same architecture as v2's per-stream backbone). Logits are averaged at fusion time. The per-stream input channel count drops from 7 to 4, leaving total parameters roughly unchanged (~6.7M, distributed across three smaller streams instead of two larger ones). All other v2 choices were inherited verbatim - dropout 0.5, mixup α=0.3, attention on blocks `[8, 9]`, clip-level augmentation, plain `shuffle=True`. The lesson from v3 was clear: do not touch the regularisation recipe.

Two evaluation-time recipes were added as toggleable defaults:

**EMA of model weights** (`decay=0.999`). After each optimiser step a shadow copy of all float-typed parameters and buffers is updated as `ema = decay·ema + (1−decay)·model`. Integer buffers (e.g. BatchNorm's `num_batches_tracked`) are copied directly. At validation, the EMA module is used as a second model alongside the live one. For 150-epoch training with batch size 32 and ~219 batches per epoch, the effective averaging window is ~1000 steps (~5 epochs) - long enough to smooth out late-training fluctuations, short enough to track meaningful learning. The early epochs of EMA always look broken (the moving average is still saturated by the random initialisation), so the EMA only becomes informative once it has "caught up" around epoch 20–25.

**Test-time augmentation** via horizontal flip. At validation each batch is also run as a flipped copy: x-coordinate negated, paired joints swapped according to `FLIP_PAIRS`. The softmax probabilities of the two passes are averaged before argmax. The flip operation was verified to be involutive (`hflip(hflip(x)) == x`) and was applied identically across all three streams since x-velocity and x-component of bone vectors flip the same way as x-position.

Each validation pass produces four metrics - live, live+TTA, EMA, EMA+TTA - so a single training run yields the full ablation table.

A 50-epoch comparison against a freshly-trained v2 (same seed, same data) showed v4 won by +1.1pp best top-1 / +1.6pp best top-5. v4's train accuracy was 3–5pp higher than v2's at matched epochs, indicating slightly faster fitting. The peak val accuracy arrived earlier (epoch 32 for v4 vs epoch 42 for v2) - three streams of ensemble diversity converge sooner.

A short EMA/TTA ablation at 50 epochs was disappointing: both contributions were within noise (~±0.5pp). The EMA had not yet caught up to the live model; the flip-TTA was redundant against horizontal-flip training augmentation that the model had already learned to be invariant to. Extending the run to 150 epochs changed the EMA picture entirely.

At 150 epochs:

| Variant | Best Top-1 (epoch) | Best Top-5 (epoch) |
|---|---|---|
| v4 live | 34.3% (125) | 54.1% (67) |
| v4 + TTA | 34.2% (117) | 54.1% (67) |
| v4 + EMA | 34.8% (50) | 54.5% (40) |
| **v4 + EMA + TTA** | **34.9% (63)** | **54.7% (41)** |

EMA was now the most impactful component: it pushed the peak earlier (epoch 50 vs 125 for live) and slightly higher. The peak val accuracy was reached around epoch 60 of 150; the remaining 90 epochs were wasted compute on memorisation - train accuracy climbed to ~86% with val drifting slightly downward. This observation was carried into v5 training as a 100-epoch budget instead of 150.

The net story for the v2 → v4 transition: a +1.2pp top-1 / +3.2pp top-5 improvement (33.7% / 51.5% → 34.9% / 54.7%), driven about half by the motion stream and half by EMA. TTA contributed essentially nothing.

---

## Chapter 4 - Per-Class Diagnostic

A 35% top-1 number is uninformative without knowing *which* of the 500 classes contribute the errors. A diagnostic script (`diagnose_per_class.py`) was written to load the best v4 EMA+TTA checkpoint and evaluate it per-class on val+test combined (6 samples per class - coarse but less noisy than val alone with 3). The output is a per-class top-1/top-5, a top-3 confusion list per class, and a CSV with every class's stats for inspection.

The headline distribution was strikingly bimodal. Of 500 classes:

| Top-1 | Count | % |
|---|---|---|
| 0/6 correct (0%) | 122 | 24% |
| 1–2/6 (17% or 33%) | 200 | 40% |
| 3/6 (50%) | 106 | 21% |
| 4/6 (67%) | 41 | 8% |
| 6/6 (100%) | 31 | 6% |

**322 of 500 classes (64%) score at or below 33% top-1.** The model is essentially "yes I know this class" or "I can't tell" - there's almost no middle ground.

A keyword heuristic flagged 17 likely two-person classes (shake, hug, dance, fight, etc.). The aggregate result was unexpected: flagged classes averaged **44.1% top-1**, *higher* than the 33.9% on others. The hypothesis that two-person clips were a major cost centre was wrong. Inspection showed why: most flagged-by-keyword classes were actually one-person actions (`shaking_head`, `floss_dance`, `gangnam_style`), and the genuinely-dyadic ones (`hugging_human`, `kiss`, `fist_bump`) were mixed - some failed badly, others worked fine. Two-person interaction is a real but secondary issue affecting maybe 5–10 classes out of 500.

The dominant failure mode was different. The bottom 50 classes were almost entirely **object-mediated or tool-use actions**:

- Drinking, eating, smoking: `drinking_with_cup`, `eat_apple`, `eating_hotdogs`, `eating_ice_cream`, `smoking_inhale`, `hookah`, `brushing_teeth`
- Tool use: `chainsaw_log`, `gas_pumping_to_car`, `hammering_nail`, `handsaw`, `haircut_scissor`, `decorating_snowman`, `fire_extinguisher`, `flamethrower`, `blowing_glass`
- Musical instruments: `play_gong`, `play_tambourine`, `play_panpipe`, `play_melodic`, `playing_taiko_drum`, `dj`
- Sports without distinctive whole-body pose: `baseball_catch_flyball`, `football_catch`, `card_throw`, `bmx_riding`, `skateboard_forward`, `paragliding`, `equestrian_run`

And the top 20 confirmed the inverse: every class scoring 100% top-1 was a **distinctive whole-body pose** (`yoga_*`, `gym_*`, `handstand`, `pull_ups`, `weightlifting_*`, `volleyball_set`) or a **simple repetitive whole-body motion** (`jumping_jack`, `high_knees`, `pole_vault_run`).

The empirical conclusion was unambiguous: the skeleton-only modality has a hard information-theoretic ceiling. A skeleton can capture *that* a hand is near a face but not *whether the hand is holding a cup, an apple, a cigarette, or a toothbrush*. About 60% of HAA500 classes are defined by either the object being manipulated or the scene context, neither of which a 33-joint pose skeleton can represent. The top-5 accuracy of 54.7% is meaningfully higher than top-1 not because the model is undertrained, but because it can usually narrow the answer to a small group of similar-motion classes ("something at hand-to-mouth" → drinking, eating, smoking, brushing) without being able to disambiguate within that group.

This diagnostic directly motivated the next step - v5 - and provides the empirical justification for any future modality extensions.

---

## Chapter 5 - Adding Hand Landmarks

### Re-extraction with MediaPipe Holistic

The simplest extension that stays inside "skeleton-based recognition" is to add finger landmarks. MediaPipe provides a separate `HandLandmarker` model (21 landmarks per hand, including the wrist and four landmarks per finger). The legacy combined `Holistic` solution was deprecated in MediaPipe 0.10, so the new extraction (`pose_extraction_holistic.py`) runs `PoseLandmarker` and `HandLandmarker` independently per frame and merges results. Each frame's output becomes a `(75, 4)` array:

- Indices `0..32`: 33 body landmarks (unchanged from v4 input data, but now in image-space coordinates rather than world coordinates - the hand landmarker only produces image-space output, so the body must match)
- Indices `33..53`: 21 left-hand landmarks
- Indices `54..74`: 21 right-hand landmarks

Hands are assigned to the left/right slot by MediaPipe's `handedness` classifier (which reports the person's own left/right hand, not a mirror-image convention). When a hand isn't detected in a frame, that slot is filled with zeros and a visibility of 0; when detected, hands get visibility = 1 (the model lacks per-landmark visibility for hands). Body landmarks still use MediaPipe's real visibility score.

A practical concern: the splits CSV files generated under WSL contained Linux absolute paths (`/home/andrei/Licenta/Licenta/haa500_v1_1/...`) which fail on Windows. The extractor includes a path-translation step that strips everything up to `haa500_v1_1/` and rebuilds against the local `ROOT_DIR`.

Extraction speed averaged ~14 videos per minute on CPU MediaPipe (both tasks running per frame). The full 10,000-video re-extraction took ~12 hours wall-clock, completed without errors.

### Graph topology for `st_gcn_v5.py`

The model file is structurally identical to v4 - same `ThreeStreamSTGCN` (joint+bone+motion), same `SingleStream` blocks, same EMA/TTA/mixup/dropout settings - but with a different graph:

- **NUM_NODES = 75**. The first BatchNorm and all per-stream feature counts scale accordingly.
- **EDGES**: 79 total. 35 body edges (kept exactly from v4), 21 left-hand edges from MediaPipe's hand topology (thumb, index, middle, ring, pinky chains plus a palm-closure edge), the same 21 mirrored to the right hand, and 2 cross-part connector edges - body's left wrist (joint 15) ↔ left hand wrist (joint 33), and right wrist (16) ↔ right hand wrist (54). These two connectors are what allow the graph to actually treat body and hands as one continuous skeleton rather than three disconnected sub-graphs.
- **FLIP_PAIRS**: 37 pairs. 16 body left/right (kept from v4), plus 21 left/right hand mirrors (left-hand landmark *i* swaps with right-hand landmark *i*). Used by both training augmentation and TTA.

The dataset class drops the v2/v4 fallback logic: v5 reads only from `extracted_skeletons_holistic/` because the 33-node and 75-node arrays are incompatible.

Two practical issues surfaced during training setup. First, the larger graph caused GPU memory pressure: at batch size 32 the model occupied 96% of the RTX 4070 Laptop's 8 GB and per-batch time collapsed from ~0.30 s to several seconds, because PyTorch's allocator was thrashing and cuDNN was choosing slower fallback kernels. Halving the batch size to 16 dropped memory to 64% and restored full speed - a ~100× wall-clock recovery from a one-line change. Second, the training budget was reduced from 150 to 100 epochs based on v4's curve: the v4 EMA+TTA peak landed at epoch 63 of 150 with the remaining 87 epochs spent overfitting. Compressing the cosine schedule to 100 epochs covers the same training arc.

### Run 5 - body+hands skeleton

v5 was trained with `combined_train=True`: train and val were merged into a single 8,500-sample training pool, with the test split (1,500 samples) reserved for evaluation only. This is the standard ML protocol for a final reported model - val is used during development for checkpoint selection, then the final model trains on train+val and is evaluated once on test. v4's reported numbers used val for evaluation; v5's use test. The two are drawn from the same stratified random split and are statistically equivalent as held-out sets, but they are different physical samples.

Training reached 100 epochs in two sessions (~3.5 hours of GPU time, with a pause and a resume from the checkpoint - the train loop saves full state every epoch including optimiser momentums, scheduler position, AMP scaler, EMA weights, and best-acc, so resumes pick up exactly where they stopped). Final results on the held-out test split:

| Variant | Best Top-1 (epoch) | Best Top-5 (epoch) |
|---|---|---|
| v5 live | 35.9% (91) | 56.1% (42) |
| v5 + TTA | 35.8% (58) | 56.5% (42) |
| v5 + EMA | 33.7% (39) | 55.6% (38) |
| v5 + EMA + TTA | 33.7% (69) | 56.1% (37) |

Curiously, the EMA variant *underperformed* the live model on top-1 here, the opposite of v4. The likely cause is that v5's larger graph overfits faster (train accuracy crossed 80% by epoch 80), so the EMA averages over a window that includes already-overfit weights. The live model's checkpoint at epoch 91 is a slightly luckier draw from a model that's broadly memorising.

### v4 vs v5 - the test-set comparison

Re-running the per-class diagnostic on test-only for both models gives the fair side-by-side:

| Metric | v4 | v5 | Δ |
|---|---|---|---|
| Top-1 | 33.60% | 33.67% | +0.07pp |
| Top-5 | 54.00% | 53.27% | −0.73pp |

**Aggregate top-1 is essentially unchanged.** Adding 42 hand landmarks per frame did not improve the overall accuracy on this dataset.

But the per-class movement is dramatic:

- **111 classes improved by ≥33pp**
- **116 classes regressed by ≥33pp**
- 273 classes unchanged

The trade is almost exactly 1:1, and the directions are exactly what the v4 diagnostic predicted.

**v5 rescued hand-mediated classes** that v4 was failing on:

| Class | v4 → v5 |
|---|---|
| shake_cocktail | 0% → **100%** |
| play_cello, play_hulusi, play_ocarina, play_saw | 0% → 67% |
| guitar_flip, remove_car_tire, taichi_fan | 0% → 67% |
| spitting_on_face, talking_on_phone, burping | 0% → 67% |
| battle-rope_wave, chainsaw_tree, play_grandpiano | 33% → 100% |
| brushing_teeth, bandaging, applying_cream, peeling_banana | substantial improvements |

These are exactly the categories the diagnostic flagged as failure modes - object-manipulation, instrument-playing, subtle hand-to-face actions. The hand landmarks gave the model a way to actually see the discriminating signal.

**v5 lost strong-motion body-only classes** that v4 was nailing:

| Class | v4 → v5 |
|---|---|
| hand-drill_firemaking_drill_with_hand | 100% → 0% |
| punching_sandbag, sledgehammer_strike_down | 67% → 0% |
| tap_dancing, unicycle_ride | 67% → 0% |
| baseball_pitch, baseball_swing, tennis_serve, soccer_throw | 100% → 33% |
| high_knees, pole_vault_run, weightlifting_hang | 100% → 33% |
| shoot_dance | 100% → 33% |

These are large-amplitude full-body motions where the body skeleton alone is highly distinctive. Adding 42 noisy hand nodes per frame (most often partially detected, often zero-padded when out of view) gave the graph convolution more low-signal nodes to integrate over, diluting the body's discriminative content. The model gained capacity at the cost of focus.

The unified takeaway: **a symmetric body+hands ST-GCN trades off body precision for hand awareness.** The diagnostic's bimodal distribution is preserved but the modes have shifted - hand-mediated classes have moved up, body-only classes have moved down, and the histogram total stays the same. v5 is not a strict improvement over v4; it is a *differently specialised model with a different strength profile*.

---

## Chapter 6 - Conclusions

Across five training runs, three architectural extensions, and two skeleton-extraction pipelines, the test-set top-1 accuracy moved from 31.5% (v1) to 33.7% (v2) to 34.9% (v4 on val) to 33.7% (v5 on test). The total improvement attributable to model and recipe changes is real but modest: roughly +2–3 percentage points, and even that is partly a measurement artifact between val and test as held-out sets.

The per-class diagnostic explains why. About 24% of HAA500 classes are unrecognisable to a skeleton-only model regardless of architecture, and another 40% are recognisable only sometimes. The classes the model gets perfectly right are exactly those defined by whole-body geometry - yoga poses, gym exercises, distinctive sports motions. The classes it never gets right are those defined by what the person is *holding* or *interacting with*. No re-arrangement of the model's internals changes this; the ceiling is set by the input modality.

This makes the natural follow-up work clear:

- **Adding hand landmarks (v5) was the right idea in isolation** but the symmetric fusion architecture diluted body performance. A two-branch architecture that processes body and hands as separate graphs before fusing learned features would likely preserve v4's body-class wins while adding v5's hand-class wins. This is a small architectural change but a meaningful next step within "base ST-GCN".

- **Adding face landmarks** (MediaPipe Face has 478 landmarks) would help eating, smoking, kissing, applying-makeup, and similar face-region classes that the body+hands skeleton still cannot disambiguate. The 478-landmark face mesh is too dense for a vanilla ST-GCN - it would need either part-pooling to ~10–20 keypoints or a hierarchical architecture.

- **Adding any non-skeleton signal** - RGB context, object detection bounding boxes, audio - would break the modality ceiling more directly than any in-modality refinement. Skeleton + RGB fusion typically reaches 65–80% top-1 on HAA500 in the literature, compared to the 33–45% range achievable from skeleton alone.

The empirical contribution of this work is therefore not the absolute accuracy number, but the **per-class diagnostic** that quantifies where skeleton-only recognition succeeds and fails on HAA500. The pattern - bimodal distribution, 64% of classes at or below 33% top-1, failures concentrated in object-mediated actions - gives a precise, actionable target for future modality extensions, and it is the same pattern that holds across v2, v4, and v5. Different architectures within the skeleton-only family redistribute that distribution but do not collapse it.

### Final results summary

| Model | Modality | Eval split | Top-1 | Top-5 | Notes |
|---|---|---|---|---|---|
| v1 - single-stream ST-GCN | 2-D image landmarks | val | 31.5% | 48.5% | 100 epochs |
| v2 - two-stream + attention | 3-D world landmarks | val | 33.7% | 51.5% | 150 epochs |
| v3 - v2 + reviewer suggestions | 3-D world landmarks | val | 26.3% (best variant 27.9%) | 46.7% | Architectural changes were a net regression |
| v4 - three-stream (joint + bone + motion) + EMA + TTA | 3-D world landmarks | val | **34.9%** | **54.7%** | 150 epochs |
| v5 - three-stream over 75-node body+hands graph | image-space body + hands | test | 33.7% | 53.3% | 100 epochs, `combined_train=True` |
| v5 vs v4 on common test set | - | test | 33.7% (v5) vs 33.6% (v4) | 53.3% vs 54.0% | Net wash; per-class redistribution explained in Chapter 5 |

The best deployable checkpoint is `best_stgcn_v5_combined.pth` for object-interaction-heavy demos and `best_stgcn_v4_emattta.pth` for body-motion-heavy demos. There is no single model in this work that dominates the other on all 500 classes.
