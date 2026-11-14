
### sigma = 0.003, L = 4, all events

| method | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| Monte Carlo (analytic) | 2.1e10 / 8.3e7 (1.00) | 2.1e12 / 4.2e9 (1.00) | 2.1e14 / 2.7e11 (1.00) | 2.1e16 / 2.0e13 (1.00) | 2.0e18 / 1.6e15 (1.00) |
| Kerdock design | 1.6e10 / 6.4e7 (1.00) | never / never (0.30) | - | - | - |
| splitting, no psi (per event) | 5.3e9 / 5.3e9 (1.00) | 1.1e10 / 1.1e10 (1.00) | 1.7e10 / 1.7e10 (1.00) | 2.7e10 / 2.7e10 (1.00) | 3.9e10 / 3.9e10 (1.00) |
| kprop K = 2 | 2.1e8 / 3.2e5 (0.51) | never / never (0.24) | never / never (0.09) | never / never (0.07) | never / never (0.06) |
| ARC's tail method (factorized K = 3) | 3.4e9 / 7.3e6 (0.96) | 3.4e9 / 7.3e6 (0.55) | never / never (0.27) | never / never (0.17) | never / never (0.09) |
| factorized K = 3, augmented | 2.9e10 / 6.3e7 (0.97) | 2.9e10 / 6.3e7 (0.57) | never / never (0.29) | never / never (0.16) | never / never (0.09) |
| MC moments + Gaussian tail | never / never (0.47) | never / never (0.21) | never / never (0.10) | never / never (0.07) | never / never (0.06) |
| MC moments + Edgeworth tail | 5.7e11 / 4.5e8 (0.98) | 5.7e11 / 4.5e8 (0.56) | never / never (0.23) | never / never (0.15) | never / never (0.07) |
| ours: MC moments + saddlepoint | 5.7e11 / 4.5e8 (0.99) | 5.7e11 / 4.5e8 (0.93) | 5.7e11 / 4.5e8 (0.86) | 5.7e11 / 4.5e8 (0.69) | 5.7e11 / 4.5e8 (0.56) |
| ours: MC moments + saddlepoint, chi mixture | 5.7e11 / 4.5e8 (1.00) | 5.7e11 / 4.5e8 (0.95) | 5.7e11 / 4.5e8 (0.84) | 5.7e11 / 4.5e8 (0.62) | never / never (0.45) |
| post hoc: saddlepoint on the K3f description | 7.9e9 / 6.2e6 (0.99) | 7.9e9 / 6.2e6 (0.86) | 7.9e9 / 6.2e6 (0.64) | never / never (0.47) | never / never (0.36) |
| post hoc: same, chi mixture | 7.9e9 / 6.3e6 (0.99) | 7.9e9 / 6.3e6 (0.84) | 7.9e9 / 6.3e6 (0.59) | never / never (0.44) | never / never (0.33) |
| post hoc: saddlepoint on the K3f augmented description | 6.8e10 / 5.3e7 (1.00) | 6.8e10 / 5.3e7 (0.92) | 6.8e10 / 5.3e7 (0.71) | 6.8e10 / 5.3e7 (0.57) | never / never (0.47) |

U3 vs best sampling/hybrid competitor (per event): 1e-4: C-GS 9.3e-3 [8.9e-3, 9.5e-3]; 1e-6: C-GS 1.8e-2 [1.7e-2, 1.9e-2]; 1e-8: C-GS 2.9e-2 [2.8e-2, 3.1e-2]; 1e-10: C-GS 4.5e-2 [4.3e-2, 4.7e-2]; 1e-12: C-GS 6.5e-2 [5.6e-2, 6.8e-2]

U3 vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.3e-1 [1.3e-1, 1.4e-1]; 1e-6: C-MCx-edge 1.0e0 [1.0e0, 1.0e0]; 1e-8: C-GS 3.8e1 [3.6e1, 3.9e1] WIN; 1e-10: C-GS 5.7e1 [5.5e1, 6.0e1] WIN; 1e-12: C-GS 8.3e1 [7.1e1, 8.7e1] WIN

U3r vs best sampling/hybrid competitor (per event): 1e-4: C-GS 9.3e-3 [8.9e-3, 9.5e-3]; 1e-6: C-GS 1.8e-2 [1.7e-2, 1.9e-2]; 1e-8: C-GS 2.9e-2 [2.8e-2, 3.1e-2]; 1e-10: C-GS 4.4e-2 [4.0e-2, 4.6e-2]; 1e-12: C-GS 0.0e0 [0.0e0, 5.6e-2]

U3r vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.3e-1 [1.3e-1, 1.4e-1]; 1e-6: C-MCx-edge 1.0e0 [1.0e0, 1.0e0]; 1e-8: C-GS 3.8e1 [3.6e1, 3.9e1] WIN; 1e-10: C-GS 5.6e1 [5.1e1, 5.9e1] WIN; 1e-12: C-GS 0.0e0 [0.0e0, 7.1e1]

### sigma = 0.003, L = 4, validated events only

| method | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| Monte Carlo (analytic) | 2.1e10 / 8.3e7 (1.00) | 2.1e12 / 4.2e9 (1.00) | 2.1e14 / 2.8e11 (1.00) | 2.1e16 / 2.0e13 (1.00) | 2.0e18 / 1.6e15 (1.00) |
| Kerdock design | 1.6e10 / 6.4e7 (1.00) | never / never (0.30) | - | - | - |
| splitting, no psi (per event) | 5.3e9 / 5.3e9 (1.00) | 1.1e10 / 1.1e10 (1.00) | 1.7e10 / 1.7e10 (1.00) | 2.7e10 / 2.7e10 (1.00) | 3.9e10 / 3.9e10 (1.00) |
| kprop K = 2 | 2.1e8 / 3.2e5 (0.51) | never / never (0.24) | never / never (0.09) | never / never (0.07) | never / never (0.06) |
| ARC's tail method (factorized K = 3) | 3.4e9 / 7.3e6 (0.96) | 3.4e9 / 7.3e6 (0.55) | never / never (0.27) | never / never (0.17) | never / never (0.09) |
| factorized K = 3, augmented | 2.9e10 / 6.3e7 (0.97) | 2.9e10 / 6.3e7 (0.57) | never / never (0.29) | never / never (0.16) | never / never (0.09) |
| MC moments + Gaussian tail | never / never (0.47) | never / never (0.21) | never / never (0.10) | never / never (0.07) | never / never (0.06) |
| MC moments + Edgeworth tail | 5.7e11 / 4.5e8 (0.98) | 5.7e11 / 4.5e8 (0.56) | never / never (0.23) | never / never (0.15) | never / never (0.07) |
| ours: MC moments + saddlepoint | 5.7e11 / 4.5e8 (0.99) | 5.7e11 / 4.5e8 (0.93) | 5.7e11 / 4.5e8 (0.86) | 5.7e11 / 4.5e8 (0.69) | 5.7e11 / 4.5e8 (0.56) |
| ours: MC moments + saddlepoint, chi mixture | 5.7e11 / 4.5e8 (1.00) | 5.7e11 / 4.5e8 (0.95) | 5.7e11 / 4.5e8 (0.84) | 5.7e11 / 4.5e8 (0.62) | never / never (0.45) |
| post hoc: saddlepoint on the K3f description | 7.9e9 / 6.2e6 (0.99) | 7.9e9 / 6.2e6 (0.86) | 7.9e9 / 6.2e6 (0.64) | never / never (0.47) | never / never (0.36) |
| post hoc: same, chi mixture | 7.9e9 / 6.3e6 (0.99) | 7.9e9 / 6.3e6 (0.84) | 7.9e9 / 6.3e6 (0.58) | never / never (0.44) | never / never (0.33) |
| post hoc: saddlepoint on the K3f augmented description | 6.8e10 / 5.3e7 (1.00) | 6.8e10 / 5.3e7 (0.92) | 6.8e10 / 5.3e7 (0.70) | 6.8e10 / 5.3e7 (0.57) | never / never (0.47) |

U3 vs best sampling/hybrid competitor (per event): 1e-4: C-GS 9.3e-3 [8.9e-3, 9.5e-3]; 1e-6: C-GS 1.8e-2 [1.7e-2, 1.9e-2]; 1e-8: C-GS 2.9e-2 [2.8e-2, 3.1e-2]; 1e-10: C-GS 4.5e-2 [4.3e-2, 4.7e-2]; 1e-12: C-GS 6.5e-2 [5.6e-2, 6.8e-2]

U3 vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.3e-1 [1.3e-1, 1.4e-1]; 1e-6: C-MCx-edge 1.0e0 [1.0e0, 1.0e0]; 1e-8: C-GS 3.8e1 [3.6e1, 3.9e1] WIN; 1e-10: C-GS 5.7e1 [5.5e1, 6.0e1] WIN; 1e-12: C-GS 8.3e1 [7.1e1, 8.7e1] WIN

U3r vs best sampling/hybrid competitor (per event): 1e-4: C-GS 9.3e-3 [8.9e-3, 9.5e-3]; 1e-6: C-GS 1.8e-2 [1.7e-2, 1.9e-2]; 1e-8: C-GS 2.9e-2 [2.8e-2, 3.1e-2]; 1e-10: C-GS 4.4e-2 [4.0e-2, 4.6e-2]; 1e-12: C-GS 0.0e0 [0.0e0, 5.6e-2]

U3r vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.3e-1 [1.3e-1, 1.4e-1]; 1e-6: C-MCx-edge 1.0e0 [1.0e0, 1.0e0]; 1e-8: C-GS 3.8e1 [3.6e1, 3.9e1] WIN; 1e-10: C-GS 5.6e1 [5.1e1, 5.9e1] WIN; 1e-12: C-GS 0.0e0 [0.0e0, 7.1e1]

### sigma = 0.003, L = 8, all events

| method | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| Monte Carlo (analytic) | 4.3e10 / 1.7e8 (1.00) | 4.3e12 / 8.5e9 (1.00) | 4.3e14 / 5.6e11 (1.00) | 4.1e16 / 4.0e13 (1.00) | 3.7e18 / 2.9e15 (1.00) |
| Kerdock design | 4.0e10 / 1.5e8 (1.00) | never / never (0.25) | - | - | - |
| splitting, no psi (per event) | 1.3e10 / 1.3e10 (1.00) | 2.7e10 / 2.7e10 (1.00) | 4.9e10 / 4.9e10 (1.00) | 8.8e10 / 8.8e10 (1.00) | 1.5e11 / 1.5e11 (1.00) |
| kprop K = 2 | never / never (0.21) | never / never (0.09) | never / never (0.06) | never / never (0.05) | never / never (0.02) |
| ARC's tail method (factorized K = 3) | 1.3e10 / 2.0e7 (0.59) | never / never (0.17) | never / never (0.04) | never / never (0.04) | never / never (0.02) |
| factorized K = 3, augmented | 1.2e11 / 1.8e8 (0.71) | never / never (0.28) | never / never (0.09) | never / never (0.04) | never / never (0.02) |
| MC moments + Gaussian tail | never / never (0.19) | never / never (0.07) | never / never (0.04) | never / never (0.03) | never / never (0.03) |
| MC moments + Edgeworth tail | 1.1e12 / 8.9e8 (0.82) | never / never (0.17) | never / never (0.06) | never / never (0.04) | never / never (0.03) |
| ours: MC moments + saddlepoint | 1.1e12 / 8.9e8 (0.89) | 1.1e12 / 8.9e8 (0.71) | never / never (0.38) | never / never (0.23) | never / never (0.16) |
| ours: MC moments + saddlepoint, chi mixture | 1.1e12 / 8.9e8 (0.93) | 1.1e12 / 8.9e8 (0.81) | 1.1e12 / 8.9e8 (0.59) | never / never (0.42) | never / never (0.33) |
| post hoc: saddlepoint on the K3f description | 2.3e10 / 1.8e7 (0.81) | 2.3e10 / 1.8e7 (0.51) | never / never (0.26) | never / never (0.17) | never / never (0.12) |
| post hoc: same, chi mixture | 2.3e10 / 1.8e7 (0.81) | 2.3e10 / 1.8e7 (0.51) | never / never (0.32) | never / never (0.21) | never / never (0.16) |
| post hoc: saddlepoint on the K3f augmented description | 2.0e11 / 1.6e8 (0.93) | 2.0e11 / 1.6e8 (0.64) | never / never (0.39) | never / never (0.28) | never / never (0.19) |

U3 vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.1e-2 [1.0e-2, 1.1e-2]; 1e-6: C-GS 2.2e-2 [2.0e-2, 2.4e-2]; 1e-8: C-GS 0.0e0 [0.0e0, 3.6e-2]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3 vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.6e-1 [1.4e-1, 1.8e-1]; 1e-6: C-MC 9.5e0 [8.9e0, 1.0e1] WIN; 1e-8: C-GS 0.0e0 [0.0e0, 4.6e1]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3r vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.1e-2 [1.0e-2, 1.1e-2]; 1e-6: C-GS 2.2e-2 [2.0e-2, 2.4e-2]; 1e-8: C-GS 3.7e-2 [3.2e-2, 4.1e-2]; 1e-10: C-GS 0.0e0 [0.0e0, 6.7e-2]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3r vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.6e-1 [1.4e-1, 1.8e-1]; 1e-6: C-MC 9.3e0 [8.6e0, 9.8e0] WIN; 1e-8: C-GS 4.7e1 [4.1e1, 5.2e1] WIN; 1e-10: C-GS 0.0e0 [0.0e0, 8.6e1]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

### sigma = 0.003, L = 8, validated events only

| method | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| Monte Carlo (analytic) | 4.3e10 / 1.7e8 (1.00) | 4.4e12 / 8.5e9 (1.00) | 4.3e14 / 5.6e11 (1.00) | 4.1e16 / 4.0e13 (1.00) | 3.8e18 / 3.0e15 (1.00) |
| Kerdock design | 4.0e10 / 1.5e8 (1.00) | never / never (0.25) | - | - | - |
| splitting, no psi (per event) | 1.3e10 / 1.3e10 (1.00) | 2.7e10 / 2.7e10 (1.00) | 4.9e10 / 4.9e10 (1.00) | 8.8e10 / 8.8e10 (1.00) | 1.4e11 / 1.4e11 (1.00) |
| kprop K = 2 | never / never (0.21) | never / never (0.09) | never / never (0.06) | never / never (0.05) | never / never (0.01) |
| ARC's tail method (factorized K = 3) | 1.3e10 / 2.0e7 (0.59) | never / never (0.18) | never / never (0.04) | never / never (0.04) | never / never (0.02) |
| factorized K = 3, augmented | 1.2e11 / 1.8e8 (0.70) | never / never (0.28) | never / never (0.09) | never / never (0.04) | never / never (0.02) |
| MC moments + Gaussian tail | never / never (0.19) | never / never (0.07) | never / never (0.04) | never / never (0.03) | never / never (0.02) |
| MC moments + Edgeworth tail | 1.1e12 / 8.9e8 (0.82) | never / never (0.17) | never / never (0.06) | never / never (0.04) | never / never (0.03) |
| ours: MC moments + saddlepoint | 1.1e12 / 8.9e8 (0.89) | 1.1e12 / 8.9e8 (0.70) | never / never (0.38) | never / never (0.23) | never / never (0.16) |
| ours: MC moments + saddlepoint, chi mixture | 1.1e12 / 8.9e8 (0.93) | 1.1e12 / 8.9e8 (0.81) | 1.1e12 / 8.9e8 (0.59) | never / never (0.43) | never / never (0.34) |
| post hoc: saddlepoint on the K3f description | 2.3e10 / 1.8e7 (0.81) | 2.3e10 / 1.8e7 (0.50) | never / never (0.26) | never / never (0.17) | never / never (0.11) |
| post hoc: same, chi mixture | 2.3e10 / 1.8e7 (0.81) | 2.3e10 / 1.8e7 (0.50) | never / never (0.32) | never / never (0.21) | never / never (0.16) |
| post hoc: saddlepoint on the K3f augmented description | 2.0e11 / 1.6e8 (0.93) | 2.0e11 / 1.6e8 (0.64) | never / never (0.39) | never / never (0.28) | never / never (0.20) |

U3 vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.1e-2 [1.0e-2, 1.1e-2]; 1e-6: C-GS 2.2e-2 [2.0e-2, 2.4e-2]; 1e-8: C-GS 0.0e0 [0.0e0, 3.6e-2]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3 vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.6e-1 [1.4e-1, 1.8e-1]; 1e-6: C-MC 9.5e0 [9.1e0, 1.0e1] WIN; 1e-8: C-GS 0.0e0 [0.0e0, 4.6e1]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3r vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.1e-2 [1.0e-2, 1.1e-2]; 1e-6: C-GS 2.2e-2 [2.0e-2, 2.4e-2]; 1e-8: C-GS 3.7e-2 [3.2e-2, 4.1e-2]; 1e-10: C-GS 0.0e0 [0.0e0, 6.7e-2]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3r vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.6e-1 [1.4e-1, 1.8e-1]; 1e-6: C-MC 9.4e0 [8.9e0, 9.8e0] WIN; 1e-8: C-GS 4.7e1 [4.1e1, 5.2e1] WIN; 1e-10: C-GS 0.0e0 [0.0e0, 8.6e1]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

### sigma = 0.003, L = 32, all events

| method | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| Monte Carlo (analytic) | 1.7e11 / 6.5e8 (1.00) | 1.6e13 / 3.1e10 (1.00) | 1.5e15 / 2.0e12 (1.00) | 1.2e17 / 1.2e14 (1.00) | 8.9e18 / 6.9e15 (1.00) |
| Kerdock design | 1.5e11 / 5.9e8 (1.00) | never / never (0.38) | - | - | - |
| splitting, no psi (per event) | 6.5e10 / 6.5e10 (1.00) | 1.7e11 / 1.7e11 (1.00) | 4.9e11 / 4.9e11 (1.00) | 1.4e12 / 1.4e12 (1.00) | 4.4e12 / 4.4e12 (1.00) |
| kprop K = 2 | never / never (0.10) | never / never (0.04) | never / never (0.07) | never / never (0.04) | never / never (0.01) |
| ARC's tail method (factorized K = 3) | never / never (0.01) | never / never (0.01) | never / never (0.00) | never / never (0.01) | never / never (0.01) |
| MC moments + Gaussian tail | never / never (0.06) | never / never (0.03) | never / never (0.02) | never / never (0.03) | never / never (0.01) |
| MC moments + Edgeworth tail | never / never (0.18) | never / never (0.03) | never / never (0.02) | never / never (0.02) | never / never (0.01) |
| ours: MC moments + saddlepoint | 4.6e12 / 3.6e9 (0.56) | never / never (0.18) | never / never (0.07) | never / never (0.04) | never / never (0.05) |
| ours: MC moments + saddlepoint, chi mixture | 4.6e12 / 3.6e9 (0.56) | never / never (0.28) | never / never (0.15) | never / never (0.07) | never / never (0.06) |
| ours: mode splitting (rule < 1) | 4.4e12 / 3.5e9 (0.66) | 1.1e13 / 8.7e9 (0.56) | 1.2e13 / 9.0e9 (0.51) | never / never (0.49) | never / never (0.45) |
| ours: mode splitting (rule <= 0.81) | 4.4e12 / 3.5e9 (0.64) | never / never (0.46) | never / never (0.39) | never / never (0.29) | never / never (0.17) |
| post hoc: saddlepoint on the K3f description | never / never (0.12) | never / never (0.04) | never / never (0.03) | never / never (0.02) | never / never (0.03) |
| post hoc: same, chi mixture | never / never (0.10) | never / never (0.03) | never / never (0.01) | never / never (0.00) | never / never (0.00) |

U3 vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.3e-2 [1.1e-2, 1.5e-2]; 1e-6: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-8: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3 vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.5e-1 [1.4e-1, 1.7e-1]; 1e-6: C-MC 0.0e0 [0.0e0, 0.0e0]; 1e-8: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3r vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.3e-2 [1.2e-2, 1.5e-2]; 1e-6: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-8: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3r vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.6e-1 [1.4e-1, 1.7e-1]; 1e-6: C-MC 0.0e0 [0.0e0, 0.0e0]; 1e-8: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U1 vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.6e-2 [1.3e-2, 1.8e-2]; 1e-6: C-GS 2.1e-2 [0.0e0, 3.0e-2]; 1e-8: C-GS 3.7e-2 [0.0e0, 4.6e-2]; 1e-10: C-GS 7.4e-2 [0.0e0, 9.7e-2]; 1e-12: C-GS 5.8e-2 [0.0e0, 1.9e-1]

U1 vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.7e-1 [1.3e-1, 2.3e-1]; 1e-6: C-MC 4.3e0 [0.0e0, 8.8e0]; 1e-8: C-GS 4.8e1 [0.0e0, 5.8e1]; 1e-10: C-GS 9.4e1 [0.0e0, 1.2e2]; 1e-12: C-GS 7.4e1 [0.0e0, 2.5e2]

U1m vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.6e-2 [1.3e-2, 1.8e-2]; 1e-6: C-GS 2.1e-2 [0.0e0, 3.0e-2]; 1e-8: C-GS 0.0e0 [0.0e0, 4.6e-2]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U1m vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.7e-1 [1.3e-1, 2.3e-1]; 1e-6: C-MC 4.3e0 [0.0e0, 8.8e0]; 1e-8: C-GS 0.0e0 [0.0e0, 5.8e1]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

### sigma = 0.003, L = 32, validated events only

| method | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 |
|---|---|---|---|---|---|
| Monte Carlo (analytic) | 1.7e11 / 6.5e8 (1.00) | 1.6e13 / 3.1e10 (1.00) | 1.4e15 / 1.9e12 (1.00) | 1.3e17 / 1.3e14 (1.00) | 1.3e19 / 1.0e16 (1.00) |
| Kerdock design | 1.5e11 / 5.9e8 (1.00) | never / never (0.38) | - | - | - |
| splitting, no psi (per event) | 6.5e10 / 6.5e10 (1.00) | 1.7e11 / 1.7e11 (1.00) | 4.3e11 / 4.3e11 (1.00) | 8.5e11 / 8.5e11 (1.00) | 1.5e12 / 1.5e12 (1.00) |
| kprop K = 2 | never / never (0.10) | never / never (0.04) | never / never (0.06) | never / never (0.04) | never / never (0.01) |
| ARC's tail method (factorized K = 3) | never / never (0.01) | never / never (0.01) | never / never (0.00) | never / never (0.00) | never / never (0.01) |
| MC moments + Gaussian tail | never / never (0.06) | never / never (0.03) | never / never (0.02) | never / never (0.01) | never / never (0.00) |
| MC moments + Edgeworth tail | never / never (0.18) | never / never (0.03) | never / never (0.01) | never / never (0.01) | never / never (0.00) |
| ours: MC moments + saddlepoint | 4.6e12 / 3.6e9 (0.56) | never / never (0.18) | never / never (0.07) | never / never (0.04) | never / never (0.01) |
| ours: MC moments + saddlepoint, chi mixture | 4.6e12 / 3.6e9 (0.57) | never / never (0.28) | never / never (0.16) | never / never (0.07) | never / never (0.05) |
| ours: mode splitting (rule < 1) | 4.4e12 / 3.5e9 (0.66) | 1.1e13 / 8.7e9 (0.57) | 1.2e13 / 9.0e9 (0.54) | 1.1e13 / 8.2e9 (0.66) | 1.1e13 / 8.9e9 (0.66) |
| ours: mode splitting (rule <= 0.81) | 4.4e12 / 3.5e9 (0.65) | never / never (0.46) | never / never (0.42) | never / never (0.43) | never / never (0.31) |
| post hoc: saddlepoint on the K3f description | never / never (0.12) | never / never (0.04) | never / never (0.03) | never / never (0.00) | never / never (0.00) |
| post hoc: same, chi mixture | never / never (0.10) | never / never (0.03) | never / never (0.01) | never / never (0.00) | never / never (0.00) |

U3 vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.3e-2 [1.1e-2, 1.5e-2]; 1e-6: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-8: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3 vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.5e-1 [1.4e-1, 1.7e-1]; 1e-6: C-MC 0.0e0 [0.0e0, 0.0e0]; 1e-8: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3r vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.3e-2 [1.2e-2, 1.5e-2]; 1e-6: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-8: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U3r vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.6e-1 [1.4e-1, 1.7e-1]; 1e-6: C-MC 0.0e0 [0.0e0, 0.0e0]; 1e-8: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-10: C-GS 0.0e0 [0.0e0, 0.0e0]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U1 vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.6e-2 [1.3e-2, 1.8e-2]; 1e-6: C-GS 2.1e-2 [0.0e0, 3.1e-2]; 1e-8: C-GS 3.7e-2 [0.0e0, 4.5e-2]; 1e-10: C-GS 7.4e-2 [0.0e0, 8.3e-2]; 1e-12: C-GS 1.0e-1 [7.6e-2, 1.5e-1]

U1 vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.7e-1 [1.3e-1, 2.3e-1]; 1e-6: C-MC 4.3e0 [0.0e0, 8.8e0]; 1e-8: C-GS 4.8e1 [0.0e0, 5.7e1]; 1e-10: C-GS 9.4e1 [0.0e0, 1.1e2]; 1e-12: C-GS 1.3e2 [9.7e1, 1.9e2] WIN

U1m vs best sampling/hybrid competitor (per event): 1e-4: C-GS 1.6e-2 [1.3e-2, 1.8e-2]; 1e-6: C-GS 2.1e-2 [0.0e0, 3.1e-2]; 1e-8: C-GS 0.0e0 [0.0e0, 4.5e-2]; 1e-10: C-GS 0.0e0 [0.0e0, 6.9e-2]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]

U1m vs best sampling/hybrid competitor (amortised): 1e-4: C-Kd 1.7e-1 [1.3e-1, 2.3e-1]; 1e-6: C-MC 4.3e0 [0.0e0, 8.8e0]; 1e-8: C-GS 0.0e0 [0.0e0, 5.7e1]; 1e-10: C-GS 0.0e0 [0.0e0, 8.8e1]; 1e-12: C-GS 0.0e0 [0.0e0, 0.0e0]
