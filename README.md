This repository is a software system containing an end-to-end Whole Slide Imaging pre-processing pipeline from 
The Cancer Genome Atlas download documents, as well as a complete implementation 
of deep learning tumor segmentation from WSI binary labels as detailed in 
"Weakly supervised multiple instance learning histopathological tumor segmentation".

## Major features
This software is entirely written in Python3 and contains two major parts:
* a tool to automatically download data from [TCGA GDC Data Portal](https://portal.gdc.cancer.gov/),
which also handles tiles extraction, background removal, and tumor label extraction.
* an end-to-end pytorch software that can train many types of common image classifier
architectures for the task of tumor segmentation on WSI based on weak binary WSI 
labels indicating the presence of tumor in each WSI.

## Installation

See [INSTALL.md](INSTALL.md). (in construction)

## Quick Start

See [GETTING_STARTED.md](GETTING_STARTED.md). (in construction)

## License

This software is released under the 
[GNU Affero General Public License v3.0 license](LICENSE).

## Citation

If you use this software or any part of this software in your research, 
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
