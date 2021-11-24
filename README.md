# OVC-MMT
Source code of AAAI-OVC.

This code repository highly depends on the research [A Visual Attention Grounding Neural Model for Multimodal Machine Translation](https://arxiv.org/abs/1808.08266) and its open-source pytorch implementation [Eurus-Holmes/VAG-NMT](https://github.com/Eurus-Holmes/VAG-NMT).

## Checkpoints

link: [[baidu]](https://pan.baidu.com/s/1KHEkKK6wKOzSmxVxkylRzQ) 

password: ovc0

## Data 

- raw data for similarity searching in scripts/raw_data: 

  link: [[baidu]](https://pan.baidu.com/s/1sw-yGQWUi9qHbyuIfU7SpQ)
  password: ovc0

- data of Multi30K and AmbiguousCOCO:

  link: too large, updating.

  NOTE: If you don't want to download the large visual object features, you might download the original Multi30K/AmbiguousCOCO dataset and extract the visual object features using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) project by yourself or using our implements (scripts/bta\_vision\_extract.ipynb) that further filtered objects with low object category probabilities.

