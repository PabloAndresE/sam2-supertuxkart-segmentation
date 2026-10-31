| name | val_iou | mIoU | mIoU_shared6 | boundary_mIoU |
|---|---|---|---|---|
| backbone_convnext_frozen | 0.7699 | 0.7429 | 0.8652 | 0.2714 |
| backbone_unet_scratch | 0.7138 | 0.7251 | 0.8460 | 0.3154 |
| split_random | 0.8626 | 0.8630 | 0.8630 | 0.4262 |
| split_temporal_block | 0.7823 | 0.7626 | 0.8897 | 0.3232 |
| split_track_holdout | 0.5146 | 0.6910 | 0.6910 | 0.1804 |

mean val mIoU 0.7286 +/- 0.1309 over 5 runs
