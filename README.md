# A Fully Probabilistic Tensor Network for Regularized Volterra System Identification

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/afrakilic/BTN_Volterra_Sys_ID/blob/main/LICENSE)


This repository contains the source code used to produce the results obtained in A Fully Probabilistic Tensor Network for Regularized Volterra System Identification submitted to [International Federation of Automatic Control (IFAC) 2026](https://ifac2026.org/). This project sets fixed random seeds to promote reproducibility. All experiments were conducted on the following computer:

- **Device**: MacBook Pro 
- **Chip**: Apple M2 Pro 
- **Memory**: 16 GB LPDDR5
- **Operating System**: macOS 15.5 

However, please note that some computations may still yield slightly different results across operating systems (e.g., macOS vs Windows), hardware architectures, or Python library versions.

This work introduces Bayesian Tensor Network Volterra kernel machines (BTN-V), extending the Bayesian Tensor Network (BTN) framework to Volterra system identification. BTN-V represents Volterra kernels via canonical polyadic decomposition, reducing model complexity from $\mathcal{O}(I^D)$ to $\mathcal{O}(DIR)$. By treating all tensor components and hyperparameters as random variables, BTN-V provides predictive uncertainty estimation at no extra cost. Sparsity-inducing hierarchical priors enable automatic rank determination and learning of fading-memory behavior directly from data, improving interpretability and avoiding overfitting. Empirical results demonstrate competitive accuracy, enhanced uncertainty quantification, and reduced computational cost.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@misc{, 
}
```

---

## Installation



Make sure you have Python **3.10.16** installed.

Install dependencies:

```bash
git https://github.com/afrakilic/BTN_Volterra_Sys_ID.git
cd BTN_Volterra_Sys_ID
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```


The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/afrakilic/BTN-Kernel-Machines/blob/main/LICENSE) file included with this repository.

---

## Author

[Afra KILIC](https://www.tudelft.nl/staff/h.a.kilic/), PhD Candidate [H.A.Kilic@tudelft.nl | hafra.kilic@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of the project Sustainable learning for Artificial Intelligence from noisy large-scale data (with project number VI.Vidi.213.017) which is financed by the Dutch Research Council (NWO).

Copyright (c) 2025 Afra Kilic.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest. 
