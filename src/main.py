"""
main.py
Numerical implementation used to compute finite-size secret key rates for a probability-shaped 16-QAM
discrete-modulation CV-QKD protocol with composable security. The SDP is formulated in PICOS and solved 
with QICS.

Reference:
  "High-rate discrete-modulated continuous-variable quantum key distribution with composable security".
"""

import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import picos

from matplotlib.colors import PowerNorm

from classical_channel import build_classical_channel, compute_pass_probability
from operators_bob import fock_ops, displaced_energy_ops
from operators_joint import build_energy_observables
from helpers import build_tau_A, compute_alpha_probs_VA
from entropy_utils import delta_w, delta_smooth, delta_leak_EC
from region_operators import build_all_region_operators


def main():
#___Parameters_______________________________________________________________________________________
    Nc = 5 
    dim_A = 16

    #channel parameters
    L = 10                              # optical fiber channel (km)
    alpha_loss = 0.2                     # dB/km (fiber loss)
    eta_t = 10 ** (-alpha_loss * L/10)  # channel transmittance

    #detector parameters
    eta_d = 1.0                           # detector efficiency 
    nu_el = 0.0                          # detection noise var --> electronic noise of the detector
    xi_wc = 0.01                          # worst-case excess noise

    beta_factor = np.sqrt(eta_t*eta_d)  # amplitude attenuation coefficient


#___Step 1. Modelling of Alice's Preparation___________________________________________________________
    levels = np.array([-3, -1, 1, 3])  

    VA_target = 2     
    nu_shape = 0.2    

    alpha0_min = 0.01
    alpha0_max = 2.0
    tol_VA = 1e-4

    best_alpha = None
    best_px = None
    best_VA = None
    best_alpha0 = None

    for _ in range(60):  
        alpha0_mid = 0.5 * (alpha0_min + alpha0_max)
        alpha_mid, px_mid, VA_mid = compute_alpha_probs_VA(alpha0_mid, levels, nu_shape)

        if VA_mid < VA_target:
            alpha0_min = alpha0_mid
        else:
            alpha0_max = alpha0_mid

        best_alpha = alpha_mid
        best_px = px_mid
        best_VA = VA_mid
        best_alpha0 = alpha0_mid

        if abs(VA_mid - VA_target) < tol_VA:
            break

    alpha0 = best_alpha0
    alpha = best_alpha
    px = best_px 

    print(f"\n[ALICE] alpha0 = {alpha0:.6f}")
    print(f"[ALICE] V_A = {best_VA:.6f} SNU (target: {VA_target})")
    print(f"[ALICE] sum(p_k) = {np.sum(px):.6f}")


    # plot: alice constelation (alpha)
    #print("[TEST] Alice constelation (α_k):")
    #for k, a in enumerate(alpha):
        #print(f"  α[{k:02d}] = {a:.4f}")
    # Plot: Alice 16-QAM constellation (geometry only)
    plt.figure(figsize=(4, 4))
    plt.scatter(
        np.real(alpha),
        np.imag(alpha),
        marker='o',
        s=70,
        facecolors='none',
        edgecolors='black',
        linewidths=1.2
    )
    plt.xlabel(r'Re$(\alpha)$')
    plt.ylabel(r'Im$(\alpha)$')
    plt.xlim(-1.7, 1.7)
    plt.ylim(-1.7, 1.7)
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


    # plot: probabilities on phase space
    #for k, pk in enumerate(px):
        #(f"  p[{k:02d}] = {pk:.6f}")
    plt.figure(figsize=(4, 4))

    norm = PowerNorm(gamma=0.6, vmin=np.min(px), vmax=np.max(px))
    sizes = 40 + 400 * (px - np.min(px)) / (np.max(px) - np.min(px))
    sc = plt.scatter(
        np.real(alpha),
        np.imag(alpha),
        c=px,
        s=sizes,
        cmap='plasma',         
        norm=norm,
        edgecolors='black',
        linewidths=0.8
    )
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$p_k$')
    cbar.ax.tick_params(labelsize=9)
    plt.xlabel(r'Re$(\alpha)$')
    plt.ylabel(r'Im$(\alpha)$')
    plt.xlim(-1.7, 1.7)
    plt.ylim(-1.7, 1.7)
    plt.grid(True, linestyle='--', linewidth=0.6, alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()




#___Step 2. Modeling of Bob’s Truncated Fock Space and Energy Observables________________________________

    fock = fock_ops(Nc)
    dim_B = fock.dim_B
    print(f"[BOB] Fock space dimension dim_B = {dim_B}")

    betas = beta_factor * alpha

    # Build displaced energy operators for each symbol
    displaced_ops = []
    for k, beta in enumerate(betas):
        disp = displaced_energy_ops(beta, fock)
        displaced_ops.append(disp)

    #print("\n[TEST] displ B operators.")
    #print("\n[TEST]", disp.n_beta)
    #print("\n[TEST]", disp.n_beta_norm)

    O1_list, O2_list = build_energy_observables(displaced_ops, dim_A)
    print(f"[BOB] energy observables: {len(O1_list)} (O1) and {len(O2_list)} (O2)")

 



 #___Step 3. Construction of Acceptance Regions and Key Mapping Operators_________________________________
    alpha0_B = beta_factor * alpha0
    Delta0 = 0.35                      
    Delta = Delta0 * beta_factor
    

    R_operators = build_all_region_operators(Nc, alpha0_B,
                                              with_postselection=True,
                                              Delta=Delta)



    sqrt_R_operators = []
    for z, R_z in enumerate(R_operators):
        R_qobj = qt.Qobj(R_z) #to operate
        sqrt_R = R_qobj.sqrtm()
        sqrt_R_operators.append(sqrt_R)
    print(f"[REGION] Computed sqrt(R_z): {len(sqrt_R_operators)} operators")


    R_sum = np.zeros_like(R_operators[0])
    for Rz in R_operators:
        R_sum += Rz
    R_perp = np.eye(dim_B) - R_sum
    R_perp = 0.5 * (R_perp + R_perp.conj().T)
    sqrt_R_perp = qt.Qobj(R_perp).sqrtm().full()
    print("[REGION] Computed sqrt(R_perp) for ⊥ outcome")


    #Kraus Operators K_z = |z⟩_R ⊗ I_A ⊗ √R_z
    dim_R = 17                                  #key register dimension 
    dim_in = dim_A * dim_B                      #dim of the input before G
    #dim_out = dim_R * dim_in                   #dim  after G

    I_A = np.eye(dim_A, dtype=complex)

    K_list = []
    for z in range(16):
        ket_z = np.zeros((dim_R, 1))            
        ket_z[z, 0] = 1.0                       #in Z position 

        sqrt_Rz = sqrt_R_operators[z].full()    #dim (nc+1)*(nc+1)
        
        K_z = np.kron(ket_z, np.kron(I_A, sqrt_Rz))
        K_list.append(K_z)


    # ⊥
    ket_perp = np.zeros((dim_R, 1), dtype=complex)
    ket_perp[16, 0] = 1.0
    K_perp = np.kron(ket_perp, np.kron(I_A, sqrt_R_perp))
    K_list.append(K_perp)

    # Z_list for pinching (projectors |z><z| ⊗ I_AB)
    I_AB = np.eye(dim_A * dim_B, dtype=complex)
    Z_list = []
    for z in range(dim_R):
        ket_z = np.zeros((dim_R, 1), dtype=complex)
        ket_z[z, 0] = 1.0
        proj_z = ket_z @ ket_z.conj().T  # |z><z|
        Z_z = np.kron(proj_z, I_AB)
        Z_list.append(Z_z)


    print(f"[KRAUS] Built {len(K_list)} Kraus operators for G")
    print(f"[KRAUS] K_z shape: {K_list[0].shape}")
    print(f"[KRAUS] Built {len(Z_list)} Kraus operators for Z (pinching)")



 #___Step 4. Construction of the SDP Constraints and Objective and Numerical Solution_______________________________

    #expected values
    xi=0.01
    n_exp = np.full(dim_A,  0.5 * eta_t * xi)   
    n2_exp = np.full(dim_A, 0.5 * eta_t * xi * (eta_t * xi + 1))  
 
    tau_A = build_tau_A(alpha, px)

    w = np.sum(px * (n2_exp - n_exp) / (Nc * (Nc + 1))) #(eq 7)
    print(f"[SDP] w (Eq. 7) = {w:.6e}\n")

    #statistical parameters
    N_total = 1e10
    k_T=0.1*N_total    
    eps_AT = 7e-11

    mu_n = np.zeros(dim_A)
    mu_n2 = np.zeros(dim_A)
    for k, disp in enumerate(displaced_ops): 
        mu_n[k] = np.sqrt((disp.n_beta_norm**2) / (2 * k_T) * np.log(2 / eps_AT))
        mu_n2[k] = np.sqrt((disp.n_beta_sq_norm**2) / (2 * k_T) * np.log(2 / eps_AT))

    dim_total = dim_A * dim_B

    print(f"[SDP] total dimension = {dim_total}")


    #constraints equation
    P = picos.Problem()

    rho = picos.HermitianVariable("rho", dim_total)

    # (g) 
    P.add_constraint(rho >> 0)

    # (a) 
    P.add_constraint(picos.trace(rho) >= 1 - w)
    P.add_constraint(picos.trace(rho) <= 1)

    # (b) 
    tau_A_mat = tau_A.full()
    tau_A_const = picos.Constant("tau_A", tau_A_mat) #como picos hace cte
    rho_A = picos.partial_trace(rho, subsystems=1, dimensions=(dim_A, dim_B))
    delta_rho = rho_A - tau_A_const
    bound_b = np.sqrt(2*w - w**2)
    P.add_constraint(0.5 * picos.NuclearNorm(delta_rho) <= bound_b)

    # (c, d, e, f) 
    for k in range(dim_A):
        Pk = px[k]

        O1_mat = O1_list[k].full()
        O2_mat = O2_list[k].full()
        
        O1_mat = 0.5 * (O1_mat + O1_mat.conj().T)
        O2_mat = 0.5 * (O2_mat + O2_mat.conj().T)
        
        O1_k = picos.Constant(f"O1_{k}", O1_mat)
        O2_k = picos.Constant(f"O2_{k}", O2_mat)

        L1_k = picos.trace(O1_k * rho) / Pk
        L2_k = picos.trace(O2_k * rho) / Pk

        # (c)
        P.add_constraint(L1_k.real <= n_exp[k] + mu_n[k])
        # (d) 
        P.add_constraint(L1_k.real >= n_exp[k] - mu_n[k] - w * displaced_ops[k].n_beta_norm)
        # (e) 
        P.add_constraint(L2_k.real <= n2_exp[k] + mu_n2[k])
        # (f)
        P.add_constraint(L2_k.real >= n2_exp[k] - mu_n2[k] - w * displaced_ops[k].n_beta_sq_norm)

    print(f"[SDP] Constraints added: {len(P.constraints)}")


    #QUICS solver setup with PICOS
    K_picos = [picos.Constant(f"K_{z}", K_list[z]) for z in range(len(K_list))] #kraus operators


    g_rho = 0
    for Kz in K_picos:
        g_rho += Kz * rho * Kz.H

    dims_output = (dim_R, dim_A, dim_B)   # (16, 16, Nc+1)

    # D( g_rho || Z(g_rho) )
    obj = picos.quantkeydist(g_rho, subsystems=0, dimensions=dims_output)

    P.set_objective("min", obj)

    #test
    print(
        f"[PICOS] "
        f"rho = {rho.shape[0]}*{rho.shape[1]} | "
        f"tau_A = {tau_A_mat.shape[0]}*{tau_A_mat.shape[1]} | "
        f"K_z = {len(K_list)} × ({K_list[0].shape[0]}*{K_list[0].shape[1]}) | "
        f"Z_z = {len(Z_list)} × ({Z_list[0].shape[0]}*{Z_list[0].shape[1]}) | "
        f"O1_k = {len(O1_list)} × ({O1_list[0].shape[0]}*{O1_list[0].shape[1]}) | "
        f"O2_k = {len(O2_list)} × ({O2_list[0].shape[0]}*{O2_list[0].shape[1]}) | "
        f"g_rho = {g_rho.shape[0]}*{g_rho.shape[1]} | "
        f"dims = {dim_R}*{dim_A}*{dim_B}"
    )


    P.solve(solver="qics", verbosity=4)

    print(f"\n[SDP] Status: {P.status}")

    if P.status != "optimal":
        print("[ERROR] SDP did not converge!")
        return

    opt_val = P.value
    rho_opt = np.array(rho.value)

    # Convert nats to bits
    H_X_given_E = opt_val / np.log(2)

    print(f"[SDP] Optimal value (nats): {opt_val:.6f}")
    print(f"[RESULT] H(X|E') = {H_X_given_E:.6f} bits")
    print(f"[RESULT] Tr(rho_bar) = {np.trace(rho_opt).real:.6f}")


#___Step 5. Finite-size Corrections to the Quantum Entropy Term_________________________________________________________
    Z_dim = dim_A
    Delta_w = delta_w(w, Z_dim)
    print("[TEST] Δ(w) =", Delta_w)

    n_key = (N_total - k_T) 
    eps_bar = 1e-11   
    rank_rhoA = dim_A   # max possible rank

    delta_eps = delta_smooth(rank_rhoA, n_key, eps_bar)
    print("[TEST] δ(ε̄) =", delta_eps)

    H_X_given_E_eff = H_X_given_E - Delta_w - delta_eps
    print("[RESULT] H(X|E') corrected =", H_X_given_E_eff)



#___Step 6. Classical Channel Induced by Discretization and Error Correction Leakage_____________________________________

    P_Z_given_X = build_classical_channel(
        alpha=alpha,
        alpha0_B=alpha0_B,
        Delta=Delta,
        eta_t=eta_t,
        eta_d=eta_d,
        xi=xi_wc,
        nu_el=nu_el
    )
    p_pass = compute_pass_probability(P_Z_given_X, px)
    
    print(f"\n[CLASSICAL CHANNEL] P(Z|X) matrix shape: {P_Z_given_X.shape}")
    print(f"[CLASSICAL CHANNEL] p_pass (post-selection) = {p_pass:.6f}")
    

    P_conclusive = P_Z_given_X[:, :16]  # shape (16, 16)
    row_sums = P_conclusive.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-15] = 1.0  # avoid division by zero
    P_Z_given_X_normalized = P_conclusive / row_sums
    
    print(f"[CLASSICAL CHANNEL] Row sums (should be ~1): {P_Z_given_X_normalized.sum(axis=1)}")



#___ Step 7. Final Key Rate__________________________________________________________________________________________

    beta_rec = 0.95
    eps_EC = 2e-11
    eps_PA = 2e-11

    delta_leak, HZ, HZ_given_X = delta_leak_EC(
        P_Z_given_X=P_Z_given_X_normalized,
        p_X=px,
        n_key=n_key,
        beta_rec=beta_rec,
        eps_EC=eps_EC,
        p_pass=p_pass
    )

    print(f"\n[KEY RATE] H(Z) = {HZ:.6f} bits")
    print(f"[KEY RATE] H(Z|X) = {HZ_given_X:.6f} bits")
    print(f"[KEY RATE] delta_leak = {delta_leak:.3f} bits")

    # Final key rate (Eq. 5)
    term_quantum = (n_key / N_total) * H_X_given_E_eff
    term_leak = delta_leak / N_total
    term_PA = (2.0 / N_total) * np.log2(1.0 / eps_PA)

    key_per_use = term_quantum - term_leak - term_PA

    print(f"\n[RESULT] term_quantum = {term_quantum:.6e} bits/use")
    print(f"[RESULT] term_leak = {term_leak:.6e} bits/use")
    print(f"[RESULT] term_PA = {term_PA:.6e} bits/use")
    print(f"[RESULT] l/N = {key_per_use:.6e} bits/use")

    # Key rate in Mbit/s
    R_rep = 1e9  # 1 GHz repetition rate
    key_rate_Mbps = key_per_use * R_rep / 1e6

    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"L = {L} km, eta_t = {eta_t:.4f}")
    print(f"H(X|E') = {H_X_given_E:.4f} bits")
    print(f"H(X|E') corrected = {H_X_given_E_eff:.4f} bits")
    print(f"Key rate = {key_rate_Mbps:.3f} Mbit/s")
    print("="*60)


if __name__ == "__main__":
    main()