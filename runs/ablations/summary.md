| name | val_iou | mIoU | mIoU_shared6 | boundary_mIoU |
|---|---|---|---|---|
| backbone_convnext_frozen | 0.7684 | 0.7394 | 0.8623 | 0.2747 |
| backbone_sam2 | 0.7746 | 0.7583 | 0.8847 | 0.3113 |
| backbone_unet_scratch | 0.7086 | 0.7048 | 0.8222 | 0.3086 |
| fpn_all | 0.7775 | 0.7601 | 0.8868 | 0.3201 |
| fpn_high_mid | 0.7372 | 0.7296 | 0.8512 | 0.2733 |
| fpn_high_only | 0.7318 | 0.6566 | 0.7661 | 0.2349 |
| fpn_low_only | 0.6973 | 0.6900 | 0.8050 | 0.1482 |
| loss_blend_30_70 | 0.7797 | 0.7623 | 0.8893 | 0.3289 |
| loss_ce_only | 0.6761 | 0.7046 | 0.8168 | 0.2233 |
| loss_dice_only | 0.7457 | 0.7324 | 0.8544 | 0.2864 |
| loss_unweighted_ce | 0.7877 | 0.7676 | 0.8933 | 0.3357 |
| no_oversampling | 0.7655 | 0.7533 | 0.8788 | 0.3097 |
| split_random | 0.8607 | 0.8699 | 0.8699 | 0.4177 |
| split_temporal_block | 0.7653 | 0.7627 | 0.8898 | 0.3256 |
| split_track_holdout | 0.5100 | 0.6791 | 0.6791 | 0.1738 |

mean val mIoU 0.7391 +/- 0.0770 over 15 runs
