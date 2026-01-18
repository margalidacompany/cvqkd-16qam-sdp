"""
operators_joint.py:
Builds the joint observables used in the SDP.
"""

import qutip as qt

def build_energy_observables(displaced_ops, dim_A):
    """
    Construct the joint observables used as part of the SDP constraints.
    Inputs:
        - displaced_ops (list[DisplacedEnergy]): displaced energy objects.
        - dim_A (int): dimension of Alice's classical label space.
    Returns:
        - (O1_list, O2_list): lists of joint observables.
    """
    if len(displaced_ops) != dim_A:
        raise ValueError("len(displaced_ops) must be = dim_A .")

    O1_list, O2_list = [], []
    for k in range(dim_A):
        #|k><k|
        ket_k = qt.basis(dim_A, k)   #all 0 dim_A nut one in k
        Pk_A = ket_k * ket_k.dag()

        n_beta_k = displaced_ops[k].n_beta
        n_beta2_k = displaced_ops[k].n_beta_sq

        # A ⊗ B
        O1_k = qt.tensor(Pk_A, n_beta_k)
        O2_k = qt.tensor(Pk_A, n_beta2_k)

        O1_list.append(O1_k)
        O2_list.append(O2_k)

    return O1_list, O2_list

