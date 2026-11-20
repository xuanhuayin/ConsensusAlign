# Data and feature preparation

## Public base features

The base manifests refer to the public dataset
[schnaus/unpaired-rosetta-embeddings](https://huggingface.co/datasets/schnaus/unpaired-rosetta-embeddings).
`configs/data_sources.json` pins a dataset commit. The download command requests
only files listed in the selected manifests, rather than the entire collection.
Feature preparation and encoder inference are outside the CPU alignment runtime.

| Manifest key | Frozen representation | Dimension |
|---|---|---:|
| dinov2_b | DINOv2 ViT-B/14, token mean | 768 |
| dinov2_g | DINOv2 ViT-g/14, token mean | 1536 |
| dinov3_7b | DINOv3 ViT-7B/16, token mean | 4096 |
| franca_cls | Franca ViT-g/14, CLS | 1536 |
| franca_mean | Franca ViT-g/14, token mean | 1536 |
| ibot_swin | iBOT Swin-T, token mean | 768 |
| ibot_b | iBOT ViT-B/16, token mean | 768 |
| mpnet | all-mpnet-base-v2 | 768 |
| qwen3_embedding | Qwen3-Embedding-8B | 4096 |
| qwen3_generative | Qwen3-8B generative token pooling | 4096 |

Franca's two entries are readouts of one checkpoint. Manifest entries are
configurations, not independent model families. Public filenames preserve the
feature release's extraction identifiers.

## Formats and row identities

Files are PyTorch `.pt` tensors or non-object NumPy `.npy` arrays. Both standard
PyTorch serialization formats are accepted. Image arrays have shape `[N,D]`.
Text arrays may also have shape `[N,T,D]` with NaN-padded missing captions.
Each item must retain at least one valid caption. Stored precision is preserved
during caption reduction, followed by float32 alignment calculations.

Paths are relative to the manifest file. A manifest has `vision` and `text`
encoder-to-file mappings. The two supported fitting layouts are:

- `paired-corpus`: every input file follows the same original corpus item order.
  A seeded permutation splits it into disjoint image and caption halves.
- `unpaired`: the files already contain their modality's fitting rows. Row counts
  may differ across modalities. Within a modality, row identity and order agree.

The loader can check shapes and values. It cannot infer whether different encoder
files refer to the same underlying items. Preserve and verify this correspondence
when preparing custom features.

The `prereduce` flag on generative-text entries reproduces the paper's
stored-precision alias convention: one caption-reduction pass precedes the
regular loader normalization. Preserve this flag when using the supplied files.

## Validation

```bash
consensusalign-data check --manifest configs/coco_train.json
```

This checks all file headers, within-modality row counts, paired-corpus row counts,
and up to 16 rows per file. Classification checks labels against the image and
class counts; retrieval checks the caption axis and image count. Full loading
validates every used embedding. Run it
before an expensive experiment. Do not overwrite source feature files while a
memory-mapped fitting or evaluation process is running.

## Downstream features

The classification evaluator expects one text row per class, or `[classes,18,D]`
for the paper's 18 prompt templates. Labels are an integer tensor in class order.
The prompts are the nine forms below followed by the same forms with `the` in
place of the indefinite article. `{}` is replaced by the stored class name.

```text
a photo of a {}.
a blurry photo of a {}.
a black and white photo of a {}.
a low contrast photo of a {}.
a high contrast photo of a {}.
a bad photo of a {}.
a good photo of a {}.
a photo of a small {}.
a photo of a big {}.
```

The Flickr30k evaluator expects image rows `[1000,D_v]` and caption rows
`[1000,5,D_l]`. All five captions in row i belong to image i. The public base
feature collection and these downstream benchmark inputs have different coverage.
Prepare downstream inputs with their original dataset/model permissions.
