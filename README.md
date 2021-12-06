# OVC-MMT
Source code of [Efficient Object-Level Visual Context Modeling for Multimodal Machine Translation: Masking Irrelevant Objects Helps Grounding](https://www.aaai.org/AAAI21Papers/AAAI-1370.WangD.pdf).

This code repository was built based on the research [A Visual Attention Grounding Neural Model for Multimodal Machine Translation](https://arxiv.org/abs/1808.08266) and its open-source pytorch implementations ([zmykevin](https://github.com/zmykevin/A-Visual-Attention-Grounding-Neural-Model) and [Eurus-Holmes](https://github.com/Eurus-Holmes/VAG-NMT)). Thanks for these efforts.

## Model Checkpoints

Link: [[baidu]](https://pan.baidu.com/s/1KHEkKK6wKOzSmxVxkylRzQ) 

Password: ovc0

## Data 

- Data of Multi30K and AmbiguousCOCO:

  Link: [[baidu]](https://pan.baidu.com/s/1Ruvrt7n8VDzqE-fjwW0m8A)

  Password: ovc0

  **< NOTE >** The object-level visual features are too large (≈12GB). It is suggested to download the original Multi30K/AmbiguousCOCO dataset and then extract the visual object features from pre-trained Faster RCNN using the [bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention) project by yourself or using our implement (scripts/bta\_vision\_extract.ipynb) that further filters objects predicted with low object category probabilities.

- Raw data for similarity searching in scripts/raw_data: 

  Link: [[baidu]](https://pan.baidu.com/s/1sw-yGQWUi9qHbyuIfU7SpQ) 

  Password: ovc0

## Reference

If you use the source codes here in your work, please cite the corresponding paper. The bibtex is listed below:

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