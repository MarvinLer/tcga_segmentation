## A collection of 6481 semi-automatically generated tumor maps for the entire snap-frozen WSI of TCGA repository for breast, kidney, and bronchus and lung locations

From our experiments, three segmentation models were extracted: one with high specificity, 
one with high sensitivity, and one with good specificity and sensitivity. These 3 models were 
ensembled using a non-parametrized decision tree.

For each WSI, the 3-headed system produces one probability for each 224 pixel-width tile at 10x 
magnification, resulting in a segmentation with granularity 112x112μm². Additionally, all tiles of WSI labelled as 
normal ones are manually put to 0.

The entire flash-frozen whole slide images from the TCGA repository for the 3 locations breast, kidney and 
bronchus and lung locations were inferred with this system, and are stored in this folder in two 
subfolders:
- the [raw](raw) subfolder contains the tile probabilities, as outputted by the 3-headed system
- the [thresholded](thresholded) subfolder holds binary segmentations using 0.3 threshold 

## License

This data is released under the 
[GNU Affero General Public License v3.0 license](LICENSE).

## Citation

If you use this data in your research (e.g. pre-training, tumor maps for radiomics), 
please use the following BibTeX entry.

```BibTeX
@misc{lerousseau2020weakly,
    title={Weakly supervised multiple instance learning histopathological tumor segmentation},
    author={Marvin Lerousseau and Maria Vakalopoulou and Marion Classe and Julien Adam and Enzo Battistella and Alexandre Carré and Théo Estienne and Théophraste Henry and Eric Deutsch and Nikos Paragios},
    year={2020},
    eprint={2004.05024},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
```

