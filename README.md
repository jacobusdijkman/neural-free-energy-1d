# Neural Free Energy Functionals Package

![DFT Visualization](DFT.gif)

This is a package for training Neural Free Energy Functionals for Classical Density Functional Theory.

## Getting Started

1. Activate your virtual environment or conda environment
2. In the parent directory `1d-neural-free-energy-training`, run:
   ```bash
   pip install .
   ```
3. In `config.py`, update the following paths:
   ```python
   local_checkpoint_path: str = "path/to/where/you/want/the/models/to/be/saved/"
   local_dataset_path: str = "/path/to/datasets/" # these are rho_dataset and c2_dataset, included under /datasets/ in this repository.
   local_datasplit_path: str = "/path/to/package/cdft_1d_package/data_split/" # these are f1_train_files.txt and f1_val_files.txt included in /1d_cdft_package/data_split/ in this repository.
   ```
4. Run the training script:
   ```bash
   python train_*.py
   ```

## Citation

If you use this code in your project, please cite our paper "Learning Neural Free Energy Functionals with Pair-Correlation-Matching":

```bibtex
@article{PhysRevLett.134.056103,
  title = {Learning Neural Free-Energy Functionals with Pair-Correlation Matching},
  author = {Dijkman, Jacobus and Dijkstra, Marjolein and van Roij, Ren\'e and Welling, Max and van de Meent, Jan-Willem and Ensing, Bernd},
  journal = {Phys. Rev. Lett.},
  volume = {134},
  issue = {5},
  pages = {056103},
  numpages = {7},
  year = {2025},
  month = {Feb},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.134.056103},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.134.056103}
}
```
