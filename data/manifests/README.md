Place author-generated CSV manifests here. Do not commit restricted dataset images.

Create one row per image with columns:
image_id,dataset,split,image_path,mask_path,partition,scene_id,curvature,curvature_tier

Do not fill curvature/tier by hand unless the procedure is documented and audited.
Keep train/tuning/threshold_validation/evaluation partitions disjoint at image level.
Where sequence identifiers exist, consider sequence-level grouping to avoid scene leakage.

The benchmark manifest is `roadsegbench_manifest_with_curvature.csv`
(SHA-256 53cbcd94881654cfe329519454d98510318742ab9113b0f0b27d0548210a6b11).
See the main README, section 2, "Frozen manifest".
