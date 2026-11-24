
### Frozen benchmark manifest

The exact frozen manifest used for the experiments is included in:

`data/manifests/roadsegbench_manifest_with_curvature.csv`

The manifest contains 31,475 image records with dataset identifiers,
partition assignments, the outline-curvature index of each road mask and its
outline-complexity tier. For historical reasons the columns are named `curvature*` and the
tier labels are `Gentle`, `Moderate` and `Sharp`; these are the **Low**, **Medium** and **High**
mask-outline complexity tiers of the revised manuscript (thresholds 0.013447 and 0.022563,
`configs/experiment.yaml`). The index describes the complexity of the road-mask outline, not
road bending.

SHA-256 checksum:

`53cbcd94881654cfe329519454d98510318742ab9113b0f0b27d0548210a6b11`

Researchers can verify the integrity of the downloaded manifest using
the SHA-256 checksum above.

The underlying dataset images and masks are not redistributed.
Researchers must obtain the datasets from their respective official providers.
