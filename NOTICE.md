# Third-party content and licensing

The **code** in this repository and the **trained decoder weights** are MIT licensed
(see [`LICENSE`](LICENSE)). The decoder was trained by the repository author and contains
no third-party weights.

## SAM 2 encoder

The frozen vision encoder is `facebook/sam2.1-hiera-tiny`, downloaded at load time from
Hugging Face and **not redistributed here**. SAM 2 is released by Meta under the Apache
2.0 licence. Refer to the model card on Hugging Face for its terms.

## ConvNeXt baseline

`torchvision`'s ImageNet ConvNeXt-Tiny weights are downloaded at load time and not
redistributed.

## Dataset: rendered SuperTuxKart frames

The `data-v1` release contains 1500 frame/mask pairs rendered from
[SuperTuxKart](https://supertuxkart.net) through
[`pystk`](https://github.com/philkr/pystk).

SuperTuxKart is free software: its **code** is GPL-3.0-or-later and its **art assets**
are predominantly CC-BY-SA 4.0, with per-asset details in the game's own
`data/licenses` directory. Frames rendered from those assets are derivative works, so
the archive is distributed **under CC-BY-SA 4.0**, separately from the MIT-licensed
code, and carries this attribution:

> Images derived from SuperTuxKart (https://supertuxkart.net), © the SuperTuxKart
> contributors, art assets licensed CC-BY-SA 4.0. Rendered via pystk for research use.
> This derivative dataset is likewise licensed CC-BY-SA 4.0.

If you reuse the dataset, keep that attribution and the share-alike terms. If you only
reuse the code or the decoder weights, MIT applies and none of the above attaches.

This is a good-faith reading of the licences by the author, not legal advice. To avoid
the question entirely, regenerate the frames yourself with `pystk` rather than using the
release.
