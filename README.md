# Na2O-Temp-Effects-SilicateGlasses-MD-BLS

This repository contains Python analysis scripts and relevant data for the study:

**"The Influence of Sodium Oxide and Temperature on the Atomic Structure and Mechanical Properties of Silicate Glasses: A Molecular Dynamics and Brillouin Light Scattering Study"**  
*Hicham Jabraoui, Thibault Charpentier, Jean-Marc Delaye, Yann Vaills*  
*Acta Materialia, 2025*

---

## 🔍 Overview

This work investigates the effects of sodium oxide (Na₂O) content and temperature on the structure and mechanical properties of silicate glasses, using classical molecular dynamics (MD) simulations and Brillouin light scattering (BLS) measurements.

This repository includes all Python scripts used to analyze simulation data and generate the figures presented in the publication. Structural features, elastic properties, Voronoi statistics, and diffusion behavior are explored in detail.

---

## 📁 Repository Contents

| Filename | Description |
|---------|-------------|
| `Density_Temps_Na2Os_Figure_2.py` | Plots glass density as a function of temperature and Na₂O content (Figure 2). |
| `elastic_constants_C11_C44_room_temperature_fig3.py` | Plots elastic constants C₁₁ and C₄₄ at room temperature (Figure 3). |
| `exp_sim_C11_C44_derivatives_vs_Na2O_fig4.py` | Comparison of experimental and simulated C₁₁, C₄₄, and their derivatives vs. Na₂O content (Figure 4). |
| `mech_properties_room_temp_fig5.py` | Plots Young’s modulus (E), bulk modulus (B), and Poisson’s ratio (ν) at room temperature (Figure 5). |
| `exp_sim_E_B_nu_derivatives_vs_Na2O_fig6.py` | Compositional trends of E, B, and ν derivatives from experiment and simulation (Figure 6). |
| `structural_properties_Qn_oxygens_CN_fig7.py` | Analyzes Qⁿ species, oxygen speciation (BO/NBO), and coordination numbers (Figure 7). |
| `Ep_O_percentage_fig8.py` | Calculates energy per oxygen as a function of NBO content (Figure 8). |
| `Si_voronoi_volume_percentage_fig9.py` | Analyzes Voronoi volume distribution for silicon atoms (Figure 9). |
| `rdf_25na2o_nbo_bo_300k_fig10.py` | Computes RDFs of BO and NBO at 25% Na₂O and 300 K (Figure 10). |
| `normalized_arrhenius_plot_fig11.py` | Generates a normalized Arrhenius plot of Na diffusion (Figure 11). |
| `na_diffusion_analysis_fig12.py` | Analyzes temperature dependence of Na diffusion coefficients (Figure 12). |
| `na_voronoi_volume_percentage_fig13.py` | Voronoi volume analysis of Na atoms (Figure 13). |

---

## 📂 Supplementary Data

All optimized glass structures for each composition and temperature, as well as mean square displacement (MSD) data at different temperatures, are provided as supplementary material. Please see the journal website for access to these files.

---

## 🛠 Requirements

To run the scripts, the following Python packages are required:

- `numpy`
- `matplotlib`
- `scipy`
- `pandas`
- `seaborn`

You can install them using:

```bash
pip install numpy matplotlib scipy pandas seaborn
````

---

## 📜 License

This repository is distributed under the [MIT License](https://opensource.org/licenses/MIT). If you use the scripts or data, please cite the corresponding publication.

---

## 🔗 Citation

Please cite the paper as:

> H. Jabraoui, T. Charpentier, J.-M. Delaye, Y. Vaills,
> *The Influence of Sodium Oxide and Temperature on the Atomic Structure and Mechanical Properties of Silicate Glasses: A Molecular Dynamics and Brillouin Light Scattering Study*,
> Acta Materialia, 2025.

---

## 📬 Contact

For questions or contributions, please contact:
**Hicham Jabraoui**
📧 [hicham.jabraoui@gmail.com](mailto:hicham.jabraoui@gmail.com)
[GitHub Profile](https://github.com/JABRAOUI)

```
