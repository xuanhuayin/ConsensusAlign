# Reproduction protocol

## Base configuration

`configs/reproduce_coco.json` declares five seeds, no anchors, uniform weights,
PCA dimension 768, 30 geometric centers, 30 initializations, 10,000-item batches,
and 100 refinement updates. All settings are explicit in the plan.

The five seeds are `734796314`, `576165995`, `2197670066`, `839703249`, and
`2584932063`. COCO train2014's 82,783 items become 41,391 fitting images and
41,392 disjoint fitting caption items. No hidden image-caption correspondence is
passed to the zero-pair bridge. COCO val2014 has 40,504 evaluation pairs and was
also used during paper development.

```bash
consensusalign-reproduce --plan configs/reproduce_coco.json --output outputs/coco
```

## Output contract and resuming

A run has the following layout:

```text
outputs/coco/
  seed_00/anchors_0000/
    model.pt
    alignment.json
    run.json
  ...
  summary.json
```

`run.json` records the seed, anchor budget, full solver settings, preprocessing
flags, feature-file hashes, artifact hashes and stage times. Local feature paths
are not written into the protocol fingerprint. Encoder block order is included.

The same command resumes completed or fitted runs only when input content and
settings agree. Completed artifacts are checked against their hashes. If inputs
change, use a new output directory, or `--overwrite` to explicitly replace runs.
A `.run.lock` prevents two writers from occupying the same run directory. After a
process crash, verify that no writer remains before removing its stale lock.

```bash
consensusalign-reproduce --plan configs/reproduce_coco.json \
  --output outputs/coco --summarize-only
```

Summary-only operation reads existing results and metadata. It checks declared
pair coverage, settings, and score-file integrity. A single seed has no sample SD
and is represented by JSON `null`, not an invented zero.

## Aggregation

For each seed, average the 21 pair-level FOSCTTM values equally. Compute the mean
and sample SD across those five seed means. Count failures over the 105 pair-seed
cells, using strictly FOSCTTM > 0.05. Per-pair seed summaries are also exported.
Do not compute the headline SD over the pooled 105 values.

The few-pair plan fixes DINOv2-B/Qwen3-generative, three seeds and ten budgets
`1,2,5,10,20,50,100,200,500,1000`. Smaller anchor subsets are prefixes of larger
ones for a fixed seed. Draws occur after the seeded disjoint split. The k anchors
are shared physical pairs across all collection members. The plan evaluates its
one target interface while fitting the same full collection.

## Evaluation protocols

- **Alignment:** row-aligned COCO evaluation pairs, independently shuffled query
  and gallery orders, FOSCTTM with half credit for exact rank ties.
- **Classification:** top-1 predictions use argmax. Eighteen class prompts are
  normalized, averaged, and normalized again. AVG10 weights the ten datasets
  equally. The fixed 12-pair subset crosses six vision configurations other than
  DINOv3 with MPNet and Qwen3-Embedding-8B.
- **Datasets:** ImageNet-1k, Food-101, Flowers102, DTD, EuroSAT, Oxford-IIIT Pets,
  Caltech101, FGVC Aircraft, STL-10 and GTSRB. Test sets are used where provided,
  ImageNet uses validation, DTD uses partition 1, and EuroSAT/Caltech101 use their
  complete evaluation collections as in the manuscript.
- **Retrieval:** Flickr30k Karpathy 1k split, five captions per image, one native
  cosine score matrix and its transpose. R@1/5/10 uses the original descending
  argsort convention, including exact ties. Every image needs a valid caption.

The library evaluates these downstream tasks through manifests. Their feature
arrays are separate inputs, not fabricated by the test suite or bundled into the
source distribution. The batch runner aggregates alignment plans.

## Cost and validation scope

Fit time starts after loading the frozen encoder features and includes pooling,
WP fitting, and native-factor construction. Feature loading and evaluation are
recorded separately. Encoder feature extraction and optional diagnostic reference
fits are additional. Full COCO needs substantially more memory and time than the
small smoke example.

The tests verify mathematical operations, supported formats, splits, anchors,
rectangular factors, frozen-coordinate reuse, tie conventions, artifact recovery
and clean package installation. They use synthetic inputs and require no external
features. The reference JSON contains manuscript results. CI does not rerun the
full paper experiments, and its success is not a claim of newly reproduced
five-seed COCO numbers.
