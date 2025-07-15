# Astrocyte-Driven Neuronal Network Synchrony

This repository contains code and data for simulating and analyzing how astrocytic gliotransmission shapes network synchrony and burst dynamics in neuron-glia co-cultures. The project integrates biophysical modeling using Brian2 with in vitro multi-electrode array (MEA) experiments to explore the developmental impact of astrocyte function, depletion, and SNARE-mediated impairment.

---

## 🧠 Overview

Astrocytes play a critical role in modulating neuronal network excitability and synchrony through calcium signaling and SNARE-dependent glutamate release. This project evaluates three experimental conditions:

1. **Control (DIV 21)** – Mature astrocytes with intact gliotransmission  
2. **FUdR-treated** – Astrocyte-depleted networks (via 5-fluorodeoxyuridine)  
3. **VAMP2 Dominant-Negative (DN)** – Astrocyte reconstitution with impaired SNARE-mediated glutamate release  

---

## 📁 Repository Structure

---

## 🧪 Simulations

- Framework: [Brian2](https://brian2.readthedocs.io/)
- Duration: 15 seconds simulated time
- Time step: 0.1 ms
- MEA grid: 32×32 electrodes
- Cell counts: 1000 excitatory neurons, 250 inhibitory neurons, up to 5500 astrocytes
- Mechanisms:
  - IP₃/Ca²⁺ intracellular astrocytic dynamics
  - Vesicular gliotransmitter release (glutamate)
  - Tripartite synapse modulation
  - Phase-locking and burst detection metrics

---

## 📊 Analysis Metrics

- Mean Interspike Interval (ISI)
- Burst count per neuron
- Network burst events (≥25% neurons firing in 75 ms)
- Phase Locking Value (PLV)
- MEA activity heatmaps

---

## 📈 Results Summary

- **Astrocyte depletion** reduced burst index by **over 90%**  
- **Dominant-negative VAMP2** astrocytes failed to restore synchrony despite restored cell density  
- **Mature astrocytes** promoted structured bursts, coherent PLV, and sustained firing

---

## ⚙️ Reproducibility

- Python 3.10  
- Brian2 v2.8.0  
- OS: macOS 15.5  
- Random seed: `np.random.seed(500)`  
- Hardware: Apple M4 Pro CPU, 24 GB RAM  
- See `methods.md` for full pipeline description and statistical tests

---

## 📜 Citation

If you use this code or model in your work, please cite:

> Amanfo et al. (2025). *Temporal Dynamics of Astrocyte-Driven Network Modulation: A Dominant-Negative Simulation Approach in Brian2*. [In preparation].

---

## 📬 Contact

For questions or collaborations, contact **Tobenna Amanfo**  
📧 amanfotobenna@gmail.com
