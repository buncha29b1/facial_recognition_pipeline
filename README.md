# Facial Recognition Pipeline (Pure PyTorch)

An end-to-end face-verification project built on a pure PyTorch stack. It benchmarks a
face-recognition pipeline on the Labeled Faces in the Wild (LFW) dataset and provides a
companion notebook that verifies whether two arbitrary images show the same person.

The pipeline is the standard detect → align → embed → compare approach:

1. Detect & align each face with `MTCNN` (`facenet-pytorch`) — landmark detection plus
   affine alignment, cropped to `3×160×160`.
2. Embed each aligned face with a pre-trained `InceptionResnetV1` backbone
   (VGGFace2 weights) into a 512-dimensional feature vector.
3. Compare two embeddings with cosine similarity on L2-normalized vectors.
4. Decide / evaluate using a threshold calibrated on LFW (the Equal Error Rate
   threshold, ≈ `0.3917`).

Everything runs in native PyTorch (`torch`, `torchvision`, `facenet-pytorch`).

## Repository contents

| File | Description |
| --- | --- |
| `facial_recognition_lfw_pipeline.ipynb` | Full benchmark: auto-downloads the LFW *pairs* set, runs the pipeline over the official evaluation pairs, plots genuine/impostor score distributions, and computes the ROC/AUC, Equal Error Rate, and optimal threshold. |
| `face_verification_two_images.ipynb` | Takes two input images and runs the identical pipeline end-to-end to return a `MATCH` / `NON-MATCH` / `BORDERLINE` verdict for the pair. |
| `requirements.txt` | Python dependencies. |
| `.gitignore` | Excludes local caches, virtual environments, and the sample images. |

## Setup

Requires Python 3.9+.

```bash
# (recommended) create and activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

On a CUDA machine, install the matching `torch` / `torchvision` build from
<https://pytorch.org> first, then run `pip install -r requirements.txt`. Pre-trained
model weights and the LFW dataset are downloaded automatically the first time you run the
notebooks (cached under `data/`, which is git-ignored).

## Usage

### Benchmark the pipeline on LFW

Open `facial_recognition_lfw_pipeline.ipynb` and run the cells top to bottom. It downloads
the LFW pairs benchmark, scores every pair, and reports the ROC AUC, Equal Error Rate, and
the calibrated decision threshold, along with score-distribution and ROC plots.

### Verify two images

Open `face_verification_two_images.ipynb`, set the two image paths near the top, and run all
cells:

```python
IMAGE_A = "path/to/first.jpg"
IMAGE_B = "path/to/second.jpg"
```

The final cell prints the cosine similarity and the verdict. A one-line helper is also
provided:

```python
verify("path/to/first.jpg", "path/to/second.jpg", show=True)
# -> {'score': 0.71, 'verdict': 'MATCH', 'same_person': True}
```

Scores within a small margin of the threshold are reported as `BORDERLINE` rather than
forced into a yes/no.

## Interpreting the threshold

The threshold `≈ 0.3917` is dataset- and model-specific — it was calibrated on LFW with
this VGGFace2 backbone. The further a score sits from the threshold, the more reliable the
decision; scores hugging the line are the system's point of maximum uncertainty and can flip
with small changes in lighting, pose, or image quality. Re-calibrate on data representative
of your deployment before using it in production, and raise the threshold for high-security
use (fewer false accepts) or lower it for convenience-first use.

## Acknowledgments

- [`facenet-pytorch`](https://github.com/timesler/facenet-pytorch) for the MTCNN detector and
  the pre-trained InceptionResnetV1 (VGGFace2) backbone.
- [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) benchmark dataset.
