# OVC-MMT
Source code of [Efficient Object-Level Visual Context Modeling for Multimodal Machine Translation: Masking Irrelevant Objects Helps Grounding](https://www.aaai.org/AAAI21Papers/AAAI-1370.WangD.pdf).

This code repository highly depends on the research [A Visual Attention Grounding Neural Model for Multimodal Machine Translation](https://arxiv.org/abs/1808.08266) and its open-source pytorch implementation [Eurus-Holmes/VAG-NMT](https://github.com/Eurus-Holmes/VAG-NMT).

## Checkpoints

link: [[baidu]](https://pan.baidu.com/s/1KHEkKK6wKOzSmxVxkylRzQ) 

password: ovc0

## Data 

- data of Multi30K and AmbiguousCOCO:

  link: too large, updating.

  NOTE:

  If you don't feel like to download the large visual object features, you might download the original Multi30K/AmbiguousCOCO dataset, and then extract the visual object features from pre-trained Faster RCNN using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) project by yourself or using our implement (scripts/bta\_vision\_extract.ipynb) that further filters objects predicted with low object category probabilities by the Faster RCNN.

- raw data for similarity searching in scripts/raw_data: 

  link: [[baidu]](https://pan.baidu.com/s/1sw-yGQWUi9qHbyuIfU7SpQ) 
  password: ovc0

## Reference

If you use the dataset included here in your work, please cite the following paper:

```
@inproceedings{wang2021efficient,
  title={Efficient Object-Level Visual Context Modeling for Multimodal Machine Translation: Masking Irrelevant Objects Helps Grounding},
  author={Wang, Dexin and Xiong, Deyi},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={35},
  number={4},
  pages={2720--2728},
  year={2021}
}
```