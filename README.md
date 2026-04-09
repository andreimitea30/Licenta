# Human Action Recognition on HAA500
## Skeleton-Based Graph Convolutional Networks

---

## Chapter 1 — Exploratory Data Analysis

The starting point was the HAA500 dataset: 10,000 YouTube video clips spread across 500 fine-grained action categories, exactly 20 clips per class. The metadata was loaded from raw `.txt` files and consolidated into a single CSV (`HAA500_consolidated_metadata.csv`) containing per-clip properties — YouTube URL, start/end timestamps, frame count, resolution, FPS, whether the camera was moving, and the number of dominant figures in the shot.

The first thing that stood out was how short the clips are. The mean duration is 2.12 seconds, with most clips falling under 4 seconds. The frame counts follow the same pattern — a mean of 59.4 frames, which conveniently landed close to the 60-frame window later chosen for training. A small number of outliers stretch to 30+ seconds (slow yoga poses, sustained holds), and a handful have fewer than 10 frames (fast cuts, detection failures).

Resolution is mostly consistent: 71.5% of clips are HD 720p landscape, with the remainder split across SD 480p and lower resolutions. Frame rates vary wildly — from 6 fps up to 30 fps.

Two properties of the dataset matter most for understanding the difficulty of the task. First, 22.9% of videos have camera motion — panning, tilting, or zooming — which means skeleton coordinates extracted in image space will shift even when the person isn't moving. Second, 16.4% of clips have more than one dominant person, which is a problem for single-person pose detectors.

After confirming all 10,000 video files were physically present, a stratified 70/15/15 split was applied (`dataset_split.py`) using `sklearn`'s `StratifiedShuffleSplit` with `random_state=42`. Stratification ensures every class contributes proportionally to each split. The result: 7,000 training videos, 1,500 validation, 1,500 test — roughly 14 training samples per class.

---

## Chapter 2 — Skeleton Extraction

### First pass: 2-D image-normalised landmarks

The original extraction (`pose_extraction.py`) used MediaPipe's `PoseLandmarker` in VIDEO mode to extract 33 body joints per frame. The output per frame was `[x, y, visibility]` — coordinates normalised to `[0, 1]` relative to the image dimensions, plus a detection confidence score. Each video was saved as a `(T, 33, 3)` NumPy array under `extracted_skeletons/{split}/{class}/{video}.npy`.

This worked but had a fundamental problem: the coordinates are in image space. If a person walks toward the camera, their joints spread out across the frame even though the motion itself is just translation. If the camera pans, every joint shifts. The 2-D image coordinates mix up actual body motion and camera behaviour, and with 22.9% of clips having moving cameras, this contaminated a significant chunk of the dataset.

### Second pass: 3-D world coordinates

A second extractor (`pose_extraction_3d.py`) was written to use `pose_world_landmarks` instead of `pose_landmarks`. MediaPipe's world landmarks are a different coordinate system entirely: expressed in metres, hip-centred (the midpoint of the two hip joints is placed at the origin), and reconstructed from a 3-D body model rather than projected from the image. Each frame yields `[x, y, z, visibility]`, saved as `(T, 33, 4)`.

The practical effect is that world coordinates don't care where the person is in the frame or what the camera is doing — they describe the body's geometry in 3-D space relative to itself. Actions like reaching forward versus reaching sideways, which look identical in a 2-D side-view projection, become distinguishable through the `z` axis.

Running MediaPipe on a headless Linux environment (WSL2 without a display server) immediately produced a crash: the shared library `libmediapipe.so` has an ELF dependency on `libGLESv2.so.2`, which doesn't exist in a headless setup. The fix required three things: creating a symlink from `libGLESv2.so.2` to `libEGL.so.1` inside the virtual environment's lib directory, prepending that directory to `LD_LIBRARY_PATH`, and setting `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGL.so.1` before Python starts. These were bundled into a shell wrapper (`run_extraction_3d.sh`) so the extraction process could run unattended.

The extractor processes one video at a time: open the video with OpenCV, read frames in a loop, convert to RGB, pass to the landmarker with a monotonically increasing millisecond timestamp (derived from frame index and FPS), and write the world landmark array to a `.npy` file. If no person is detected in a frame, a zero row is inserted. If world landmarks aren't available but image landmarks are, the extractor falls back to 2-D with `z=0`. The full 10,000-video extraction ran overnight at ~42 files per minute and completed without errors — zero NaN or Inf values in any of the saved arrays.

### Preprocessing applied at training time

Rather than baking preprocessing into the `.npy` files, it's applied on-the-fly in the dataset class:

**Normalisation.** For 3-D world coordinates, MediaPipe already handles hip-centring. An additional scale normalisation divides all position values by the mean distance between shoulder joints (joints 11 and 12) across frames. This makes bodies of different physical sizes produce consistent coordinate magnitudes. For 2-D coordinates, hip-centring is applied manually first.

**Temporal sampling.** Every sequence is brought to exactly 60 frames. For longer videos, training uses a random contiguous crop of 60 frames (this is itself a form of augmentation — each epoch sees a different window). Validation and test use uniform sub-sampling across the full sequence. Videos shorter than 60 frames are padded by repeating the last frame.

**Velocity features.** Per-frame velocity is appended as extra channels: `Δx, Δy, Δz` computed as the frame-to-frame difference in joint positions. The first frame gets zero velocity. This brings each joint's feature vector to 7 dimensions: `[x, y, z, visibility, Δx, Δy, Δz]`.

**Bone features.** A second representation is derived from the same data. For each joint, the mean of all vectors pointing from that joint to its connected neighbours is computed across the 34 skeleton edges. This encodes the relative geometry of limb segments — effectively joint angles and bone lengths — which is complementary to absolute joint positions.

---

## Chapter 3 — Model Training

### Architecture: the skeleton graph

Both models treat the skeleton as a graph: 33 nodes (joints) connected by 34 edges following the anatomical structure MediaPipe defines — neck, shoulders to elbows to wrists, hips to knees to ankles, and so on. The normalised adjacency matrix is pre-computed once as `D^{-0.5}(A+I)D^{-0.5}` and stored as a fixed buffer. Each graph convolution layer adds a learnable residual `ΔA` on top of this, initialised to zero, so the model starts from the anatomical graph and can learn task-specific connections during training.

### Run 1 — Baseline single-stream ST-GCN (2-D landmarks)

The first model (`st_gcn_pipeline.py`) is a single-stream ST-GCN operating on 2-D image-normalised landmarks. The input tensor is `(batch, 7, 60, 33, 1)` — 7 channels (x, y, visibility, velocity, acceleration), 60 frames, 33 joints, 1 person. Ten blocks process the sequence with channel progression `64→64→64→64→128→128→128→256→256→256`, with stride-2 temporal downsampling at blocks 5 and 8. Each block applies a graph convolution, batch norm, ReLU, then a temporal convolution with kernel size 9. Residual connections run through every block.

The first training attempts without regularisation were textbook overfitting: training accuracy hit 99% by epoch 20 while validation stayed below 15%. The model was simply memorising the 14 training samples per class.

Several regularisation techniques were stacked until the gap closed to a manageable level:

- **Dropout (0.5)** after each temporal convolution.
- **Stochastic depth**: each block has a probability of being entirely skipped during training, increasing linearly from 0 at block 1 to 0.3 at block 10. This forces the model to learn redundant representations across blocks.
- **Label smoothing (ε=0.15)**: instead of training against a hard one-hot target, the correct class gets probability `0.85 + 0.15/500` and all others share the remaining `0.15`. Prevents the model from becoming overconfident.
- **Mixup (α=0.2)**: two training samples are blended — both the skeleton sequences and their labels — with a mixing coefficient drawn from a Beta distribution. The model must predict an interpolated label for an interpolated input. This was the single most effective regulariser: it directly attacks the memorisation problem by ensuring no training sample ever appears exactly the same twice, and by smoothing the decision boundaries between the 500 classes.

The optimiser is AdamW with weight decay `5e-4`. The learning rate follows a 5-epoch linear warm-up from `0.1×lr` to `lr=1e-3`, then cosine annealing down to `1e-5` over the remaining epochs. Gradient clipping at norm 1.0 prevents instability.

Training for 100 epochs reached **31.5% top-1 / 48.5% top-5** validation accuracy (best at epoch 86). Training accuracy was ~97%, so meaningful overfitting remained.

### Run 2 — Two-stream ST-GCN v2 (3-D landmarks)

The second model (`st_gcn_v2.py`) made four structural changes:

**3-D world coordinates as input.** Each joint's feature vector is `[x, y, z, visibility, Δx, Δy, Δz]` — the same 7 channels but with genuine depth information and without camera-motion contamination.

**Two streams.** Two independent copies of the ST-GCN run in parallel: one on joint features (absolute positions), one on bone features (relative limb geometry). Their output logit vectors are averaged. This is essentially an ensemble of two classifiers that see complementary views of the same action.

**Multi-scale temporal convolution.** The single-kernel-9 temporal convolution in each block is replaced with four parallel branches: kernels 3, 7, 9, and a max-pool branch. Each branch produces one quarter of the output channels; they are concatenated and batch-normalised back to the original channel count. Different actions unfold at different temporal scales — a finger snap takes 3 frames, a squat takes 30 — and the multi-scale branches capture all of them simultaneously.

**Temporal self-attention in the top two blocks.** The deepest blocks (256 channels) apply multi-head self-attention (8 heads) over the time axis, independently for each joint. The feature tensor `(N, C, T, V)` is reshaped to `(N×V, T, C)`, attention is applied, and the result is added back via a residual with LayerNorm. This allows the model to correlate frames that are far apart in time — useful for cyclic actions where the start and end of a cycle mirror each other.

**The NaN bug.** The first training run with this architecture appeared to work for 5 epochs, then the loss became NaN at epoch 6 and the model weights were fully corrupted by epoch 12, spending the remaining 138 epochs at 0.2% accuracy (random chance). The cause: `nn.MultiheadAttention` running inside PyTorch's AMP (automatic mixed precision, float16) causes softmax overflow — the attention scores become too large for float16 to represent, producing NaN that propagates into the weights. The GradScaler is supposed to catch this, but the NaN was occurring in the forward pass before the backward pass, so the weights were being updated with NaN values before the scaler had a chance to intervene. The fix was to force the attention computation to stay in float32 even when the rest of the model runs in float16, by wrapping it in `torch.amp.autocast(enabled=False)` and casting the input to `.float()` before entering the attention module.

After the fix, the second run trained cleanly for 150 epochs, reaching **33.7% top-1 / 51.5% top-5** — a +2.2% improvement over the baseline.

### Results

| Model | Landmarks | Top-1 | Top-5 |
|---|---|---|---|
| Run 1 — single-stream, 100 epochs | 2-D image | 31.5% | 48.5% |
| Run 2 — two-stream + attention, 150 epochs | 3-D world | **33.7%** | **51.5%** |

The ~33% ceiling is a property of the modality, not the model. A meaningful fraction of HAA500 classes can only be distinguished by what the person is holding or where they are — information that isn't present in skeleton coordinates regardless of how good the model is.
