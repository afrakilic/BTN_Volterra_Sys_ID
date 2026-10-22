# A Fully Probabilistic Tensor Network for Regularized Volterra System Identification

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/afrakilic/BTN_Volterra_Sys_ID/blob/main/LICENSE)

Code for the paper [A Fully Probabilistic Tensor Network for Regularized Volterra System Identification](https://arxiv.org/abs/2511.20457) (submitted to [IFAC 2026](https://ifac2026.org/)).

BTN-V (Bayesian Tensor Network Volterra kernel machines) represents the Volterra kernel as a
CP-decomposed tensor, which reduces the number of parameters from O(I^D) to O(DIR). All tensor
factors and hyperparameters are random variables, so the model gives predictive uncertainty at no
extra cost. Sparsity-inducing priors choose the rank and learn the fading memory of the system
from the data.

## Installation

Python 3.11 was used.

```bash
git clone https://github.com/afrakilic/BTN_Volterra_Sys_ID.git
cd BTN_Volterra_Sys_ID
pip install -r requirements.txt
```

## Repository structure

```
btnv/          the model
  model.py       BTNV class: train, predict
  utils.py       Volterra feature map and linear-algebra helpers
experiments/
  cascaded_tanks.py   reproduces Table 1, Figure 2 and Figure 3
data/
  dataBenchmark.mat      Cascaded Tanks benchmark (Schoukens et al., 2016)
  BMVALS_all_runs.csv    BMVALS predictions (mean and variance) for 10 runs
```

## Usage

```python
from btnv import BTNV

model = BTNV().train(u_train, y_train, M=100, D=3, R=20,
                     a0=1e-3, b0=1e-3, c0=1e-5, d0=1e-6, g0=1e-6, h0=1e-6, max_iter=25)
mean, std = model.predict(u_test)
print(model.rank, model.train_time)
```

## Reproducing the paper

```bash
python experiments/cascaded_tanks.py
```

This trains BTN-V for 10 seeds with the lag precisions δ on and off, prints Table 1 (RMSE, NLL,
training time and effective rank, plus the BMVALS baseline) and shows Figures 2 and 3.
The RVS and SED-MPK rows of Table 1 are taken from the literature, and the BMVALS training time was
measured with the original MATLAB code.

All experiments use fixed random seeds. They were run on a MacBook Pro (Apple M2 Pro, 16 GB,
macOS 15.7.1); other operating systems, hardware or library versions may give slightly different
numbers, and training times depend on the machine.

## Citation

```bibtex
@misc{kilic2025fullyprobabilistictensornetwork,
      title={A Fully Probabilistic Tensor Network for Regularized Volterra System Identification},
      author={Afra Kilic and Kim Batselier},
      year={2025},
      eprint={2511.20457},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2511.20457},
}
```

## License and author

GNU General Public License, see [LICENSE](LICENSE).

[Afra Kilic](https://www.tudelft.nl/staff/h.a.kilic/), PhD candidate, [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/), Delft University of Technology (H.A.Kilic@tudelft.nl | hafra.kilic@gmail.com).
This research is part of the project Sustainable learning for Artificial Intelligence from noisy large-scale data (project number VI.Vidi.213.017), financed by the Dutch Research Council (NWO).

Copyright (c) 2025 Afra Kilic. Technische Universiteit Delft hereby disclaims all copyright interest.
