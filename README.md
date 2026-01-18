# cvqkd-16qam-sdp
This repository contains the numerical implementation developed as part of a Master's thesis to compute finite-size secret key rates for a probability-shaped 16-QAM
discrete-modulation continuous-variable quantum key distribution (CV-QKD) protocol with composable security.

The security analysis follows the framework of   “High-rate discrete-modulated continuous-variable quantum key distribution with composable security” (arXiv:2503.14871),  and is formulated as a semidefinite program (SDP) using PICOS, solved with the QICS solver.

---

## Overview

The code implements the following steps:

1. Modelling of Alice’s Preparation.
2. Modeling of Bob’s Truncated Fock Space and Energy Observables.
3. Construction of Acceptance Regions and Key Mapping Operators.
4. Construction of the SDP Constraints and Objective and Numerical Solution.
5. Finite-size Corrections to the Quantum Entropy Term.
6. Classical Channel Induced by Discretization and Error Correction Leakage.
7. Final Key Rate.

The implementation is fully numerical and is intended to reproduce and study the key-rate of the protocol under realistic finite-size assumptions.

---

## Repository structure

```text
src/
├── main.py                 # Main script reproducing the key-rate computation
├── operators_bob.py        # Bob's truncated Fock space and displaced energy operators
├── operators_joint.py      # Joint Alice–Bob observables
├── region_operators.py     # Decision-region operators for heterodyne detection
├── classical_channel.py    # Classical channel induced by discretization
├── entropy_utils.py        # Entropy and finite-size correction utilities
└── helpers.py              # Alice-side helper functions
```

---
## Usage

```bash
python main.py
```

---

## Reference
This implementation follows the security analysis presented in  “High-rate discrete-modulated continuous-variable quantum key distribution with composable security”,  M. Wu *et al.*, arXiv:2503.14871 (2025).

### BibTeX

```bibtex
@misc{wu2025highratediscretemodulatedcontinuousvariablequantum,
      title={High-rate discrete-modulated continuous-variable quantum key distribution with composable security}, 
      author={Mingze Wu and Yan Pan and Junhui Li and Heng Wang and Lu Fan and Yun Shao and Yang Li and Wei Huang and Song Yu and Bingjie Xu and Yichen Zhang},
      year={2025},
      eprint={2503.14871},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2503.14871}, 
}
```

---

## Limitations

- The semidefinite program scales poorly with the photon-number cutoff and constellation size. In particular, the dimension of the optimization variable grows as  
  \( \dim(\rho) = \text{dim}_A \times (\text{Nc}+1) \), which leads to high memory usage.

- For realistic parameter choices, the SDP may require access to high-performance computing resources.

