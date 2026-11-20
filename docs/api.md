# CLI and Python API

Install the package before running examples. `consensusalign --help` lists the
fit, evaluate, attach, adapt, export, references and diagnose commands.

## Python fitting

```python
from consensusalign import ConsensusAlign, WPConfig, load_features

vision = {"image_encoder": load_features("image_fit.pt")}
text = {"text_encoder": load_features("text_fit.pt")}
model = ConsensusAlign.fit(vision, text, latent_dim=768, wp=WPConfig(num_workers=1))
model.save("model.pt")
```

These files already contain separate unpaired fitting sets, aligned across
encoders within each modality. `load_features` supplies normalized float32 rows.
The Python API does not split the rows again. For a paired corpus use the CLI
manifest layout or `consensusalign.data.fitting_data` to make the exact split.
Use `num_workers=1` in interactive sessions. Standalone scripts using multiple
workers need an `if __name__ == "__main__":` entry guard.

## Fit a collection or anchors

```bash
consensusalign fit --manifest configs/coco_train.json --seed-index 0 \
  --output outputs/model.pt
consensusalign fit --manifest collection.json --weighting agreement \
  --output outputs/weighted.pt
consensusalign fit --manifest configs/coco_train.json --anchors 10 \
  --output outputs/anchors.pt
```

The shared dimension and WP numerical budgets are available in `fit --help`.
Defaults are the paper configuration. A larger heterogeneous collection requires
its own aligned feature manifest. Adding CLIP towers follows the same API but
introduces their paired-pretraining resource.

## Attach, adapt, export

```bash
consensusalign attach --model outputs/model.pt --modality vision \
  --name new_encoder --features new_encoder_source.pt --output outputs/attached.pt
consensusalign adapt --model outputs/model.pt --manifest target.json \
  --output outputs/adapted.pt
consensusalign export --model outputs/model.pt --vision dinov2_b --text mpnet \
  --output outputs/interface.pt
```

For a CLI paired-corpus checkpoint, attachment accepts features of the full source
corpus and selects the stored fitting rows. For a direct Python/unpaired fit,
provide the same fitting rows in the same order as the cached pooled coordinates.
Adaptation needs every source encoder under its original name. It may also
receive additional encoders to calibrate on the target items.

```python
import torch
from consensusalign import NativeMap, load_features

saved = torch.load("outputs/interface.pt", weights_only=True)
interface = NativeMap(saved["mean_x"], saved["mean_y"], saved["weight"])
scores = interface.similarity(load_features("image_queries.pt"), load_features("texts.pt"))
```

## Evaluate

`consensusalign evaluate --model ... --manifest ... --output ...` accepts a
manifest with `task` set to `alignment`, `classification`, or `retrieval`.
Encoder names must match the fitted model. Classification adds a `labels` path.
For example:

```json
{
  "task": "classification",
  "vision": {"dinov2_b": "images.pt"},
  "text": {"mpnet": "class_prompts.pt"},
  "labels": "labels.pt"
}
```

For retrieval use the same structure without labels and set `task` to `retrieval`.
Keep all captions per image in a three-dimensional text array. Alignment uses
row-aligned evaluation pairs. Evaluation data is separate from alignment fitting.
Metric values are fractions. Multiply accuracy and recall by 100 for percentages.

## Optional agreement

```bash
consensusalign references --manifest configs/coco_train.json --seed-index 0 \
  --output outputs/references.pt
consensusalign diagnose --model outputs/model.pt --manifest configs/coco_train.json \
  --references outputs/references.pt --output outputs/agreement.json
```

Use the same corpus, encoder collection, split and seed for references and the
candidate. The score measures support from other configurations. It is additional
to alignment and does not produce a universally calibrated failure threshold.
