This repository is a software system containing an end-to-end Whole Slide Imaging pre-processing pipeline from 
The Cancer Genome Atlas download documents, as well as a complete implementation 
of deep learning tumor segmentation from WSI binary labels as detailed in 
"Weakly supervised multiple instance learning histopathological tumor segmentation".

<div align="center">
  <img alt="Example of WSI segmentations" src="img/example.gif" />
  <p>2 examples of Whole Slide Image tumor segmentation (black background; blue: normal tissue; pink: neoplastic tissue).</p>
</div>

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
@misc{lerousseau2020weaklyseg,
  author =       {Marvin Lerousseau, Maria Vakalopoulou, Marion Classe, 
                  Julien Adam, Enzo Battistella, Alexandre Carr\'{e}, 
                  Th\'{e}o Estienne, Th\'{e}ophraste Henry, Eric Deutsch 
                  and Nikos Paragios},
  title =        {Weakly supervised multiple instance learning 
                  histopathological tumor segmentation},
  year =         {2020}
}
```
