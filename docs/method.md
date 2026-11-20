# Method

## Input contract

Each modality contains row-aligned frozen features: every image encoder sees
identical image item IDs, and every text encoder sees identical text item IDs.
The two fitting sets contain no matching image-caption pair. Dimensions may
differ across encoders and the two fitting row counts may differ.

The loader normalizes raw vectors. Multiple captions are normalized individually,
averaged over valid captions, and normalized again in stored precision. Each
encoder estimates its fitting mean, subtracts it and normalizes rows.

## Pool, fit, distribute

For image features X_v, concatenate `w_v X_v / sqrt(M)` in the declared encoder
order. Subtract the concatenation's mean and retain the leading d right singular
vectors of a full thin SVD. The PCA coordinates are unwhitened. Repeat on text.
The default dimension is d = 768 and the base pool uses unit weights.

WP centers and row-normalizes these two pooled feature sets, initializes from
30 geometric centers over 30 initializations, and alternates Hungarian assignments
and polar updates. Its default batches contain 10,000 rows and it performs
100 refinement steps. Its numerical seed is the run seed plus seven.

The native factors use the **raw PCA coordinates**, not WP-normalized latents:

```text
Q_v = polar(X_v^T Z_V)
R_l = polar(Y_l^T Z_L)
W_vl = Q_v C R_l^T
```

`polar(A) = U V^T` for a full thin SVD. It supports rectangular native dimensions,
including dimensions smaller than d. Inference centers native vectors with their
stored fitting means, normalizes them, and scores `cos(x_hat W_vl, y_hat)`.
This is the deployed native score used by the package.

## Weights and anchors

Agreement weighting computes linear CKA on up to 8,000 shared fitting rows,
selected with generator seed zero. Each member's mean CKA with other members is
divided by the **upper median** member agreement. The amplitude weight is
`min(1, agreement / median)^2`. Its contribution to the concatenated Gram matrix
is squared again. Fewer than three members use unit weights.

Paired anchors use the fitting native means and frozen PCA projections. They
enter WP initialization and every polar update, weighted by the matched unpaired
batch size divided by the anchor count. Every encoder shares the same physical
anchor IDs. This is separate from the zero-pair setting.

## Reuse

Attachment fits a new native factor against the cached source pooled coordinates.
It keeps the original pool membership, PCA and bridge fixed. Corpus adaptation
keeps source weights, bases, concatenation offsets and C, computes target native
means, and refits native factors in those source coordinates. It requires target
features from every source-pool encoder. Exporting one interface retains only its
two native fitting means and matrix.

## Optional diagnostic

Independent pairwise references use 30 initialization restarts, batches of 5,000,
and 30 refinement updates. A target's reference set shares neither of its exact
endpoint configurations. The base 7×3 pool fits 21 references and uses 12 per target.
Scores are raw bilinear products on shared unpaired batches of 5,000 rows with
seed plus five. Hungarian assignment agreement is returned as a score.

References can share model families and the score's scale varies by corpus.
The method does not infer a universal pass/fail threshold. Reference fitting and
scoring costs are additional to pooled alignment.
