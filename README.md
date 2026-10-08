# 📡 AI 6G RIS Platform

> An **AI-powered dashboard for visualising 6G communication performance** with **Reconfigurable Intelligent Surfaces (RIS)**. The platform turns wireless-communication (ECE) theory — channel models, SNR, capacity, outage, BER, beamforming — into live, interactive graphs.

![Domain](https://img.shields.io/badge/Domain-6G%20Wireless-blue)
![Tech](https://img.shields.io/badge/Tech-RIS%20%7C%20AI%2FML-purple)
![Status](https://img.shields.io/badge/Status-Research%20Project-green)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [What Is a RIS?](#-what-is-a-ris)
- [Features](#-features)
- [Dashboard Graphs & the Theory Behind Them](#-dashboard-graphs--the-theory-behind-them)
- [Mathematical Models & Formulas](#-mathematical-models--formulas)
- [AI / ML Models](#-ai--ml-models)
- [System Architecture](#️-system-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#️-getting-started)
- [Simulation Parameters](#-simulation-parameters)
- [Assumptions & Limitations](#-assumptions--limitations)
- [Future Improvements](#-future-improvements)
- [References](#-references)
- [Author](#-author)

---

## 🎯 Overview

Sixth-generation (6G) networks aim for extreme data rates, ultra-low latency and massive connectivity, often using millimetre-wave and terahertz bands where signals are easily blocked. A **Reconfigurable Intelligent Surface (RIS)** helps by reflecting signals in a controlled direction, effectively *programming* the wireless environment.

**AI 6G RIS Platform** lets a user:

1. Choose system parameters (frequency, distance, number of RIS elements, phase resolution, channel type …)
2. Run a physics-based simulation of the BS → RIS → user link
3. Apply an **AI model** to optimise the RIS phase shifts
4. See the resulting **performance graphs** on a dashboard, side by side with theoretical baselines

---

## 🧠 What Is a RIS?

A RIS is a planar surface made of many low-cost, nearly passive **unit cells**. Each unit cell applies a controllable **phase shift** (and sometimes amplitude change) to the incident wave before re-radiating it.

```text
                    ┌──────────────────────────┐
   Base Station ───►│  RIS  (N unit cells)     │───► User
        (BS)    g   │  θ₁  θ₂  θ₃ ... θN       │  h_r
                    └──────────────────────────┘
         ╲                                       ╱
          ╲_____________ direct link h_d ________╱   (may be blocked)
```

- **Passive reflection** — no RF chains or power amplifiers, so low power consumption
- **Programmable** — phases are set electronically (PIN diodes / varactors)
- **Purpose** — extend coverage, bypass blockage, boost SNR, improve spectral & energy efficiency

---

## 🚀 Features

- 📊 Interactive dashboard of 6G link performance
- 🧮 Physics-based channel and signal model for RIS-assisted links
- 🤖 AI-driven RIS phase-shift optimisation, compared against baselines
- 🎛️ Adjustable parameters: number of elements, phase bits, frequency, distance, transmit power, bandwidth
- 📈 Theory vs. simulation overlays for validation
- 🔍 Side-by-side comparison: **with RIS** vs. **without RIS** vs. **random phase**

<!-- TODO: confirm and trim this list to match what the dashboard actually implements. -->

---

## 📊 Dashboard Graphs & the Theory Behind Them

| Graph | X-axis → Y-axis | ECE concept it demonstrates | Key formula |
| ----- | --------------- | --------------------------- | ----------- |
| **SNR vs. RIS elements** | N → SNR (dB) | Coherent combining gives ~N² power gain | $\text{SNR}\propto N^2$ |
| **Achievable rate vs. SNR** | SNR (dB) → bits/s/Hz | Shannon capacity | $R=\log_2(1+\gamma)$ |
| **Rate vs. distance** | d (m) → Mbps | Path loss and cascaded RIS link | $PL \propto d_1^2 d_2^2$ |
| **Outage probability vs. SNR** | SNR (dB) → $P_{out}$ | Fading and reliability | $P_{out}=\Pr(\gamma<\gamma_{th})$ |
| **BER vs. SNR** | SNR (dB) → BER | Modulation performance | $Q(\sqrt{2\gamma})$ for BPSK |
| **Phase-quantisation loss** | bits → dB loss | Finite-resolution unit cells | $\text{sinc}^2(\pi/2^b)$ |
| **Energy efficiency** | N → bit/Joule | Passive RIS vs. active relays | $EE=R/P_{total}$ |
| **Beam pattern** | angle → gain (dB) | Generalised reflection law | $\sin\theta_r-\sin\theta_i=\tfrac{\lambda}{2\pi}\tfrac{d\Phi}{dx}$ |
| **AI training curve** | epoch → loss / reward | Learning convergence | MSE / cumulative reward |

<!-- TODO: match this table to the real graphs in the dashboard. -->

---

## 📐 Mathematical Models & Formulas

### 1. Propagation & path loss

**Friis free-space equation**

$$P_r = P_t\,G_t\,G_r\left(\frac{\lambda}{4\pi d}\right)^2,\qquad \lambda=\frac{c}{f}$$

**Free-space path loss (dB)**

$$PL(\text{dB}) = 32.45 + 20\log_{10}d_{\text{km}} + 20\log_{10}f_{\text{MHz}}$$

**Far-field (Fraunhofer) distance** of an aperture of largest dimension $D$:

$$d_F = \frac{2D^2}{\lambda}$$

**Doppler shift** for a user moving at speed $v$:

$$f_d = \frac{v f_c}{c}$$

### 2. Noise

$$N_0 = kT\quad\Rightarrow\quad N(\text{dBm}) = -174 + 10\log_{10}B + NF$$

where $B$ is the bandwidth in Hz and $NF$ is the receiver noise figure.

### 3. Channel models

**Rayleigh fading** (rich scattering, no line-of-sight): $h\sim\mathcal{CN}(0,\sigma^2)$

**Rician fading** with K-factor $K$ (line-of-sight present):

$$h=\sqrt{\frac{K}{K+1}}\,h_{LOS}+\sqrt{\frac{1}{K+1}}\,h_{NLOS}$$

### 4. RIS-assisted signal model

Let

- $g\in\mathbb{C}^{N\times1}$ — BS → RIS channel
- $h_r\in\mathbb{C}^{N\times1}$ — RIS → user channel
- $h_d$ — direct BS → user channel
- $\boldsymbol{\Theta}=\mathrm{diag}\big(\beta_1e^{j\theta_1},\dots,\beta_Ne^{j\theta_N}\big)$ — RIS reflection matrix

The **effective channel** and received signal are

$$h_{eff}=h_d+h_r^{H}\boldsymbol{\Theta}\,g,\qquad y=\sqrt{P_t}\,h_{eff}\,s+n,\quad n\sim\mathcal{CN}(0,\sigma^2)$$

The **received SNR** is

$$\gamma=\frac{P_t\,|h_{eff}|^2}{\sigma^2}$$

### 5. Optimal (co-phasing) RIS configuration

To make all reflected paths add *in phase* with the direct path, choose

$$\theta_n^{\star}=\arg(h_d)-\arg(h_{r,n})-\arg(g_n)$$

With $\beta_n=1$ the maximum channel magnitude is

$$|h_{eff}|_{\max}=|h_d|+\sum_{n=1}^{N}|h_{r,n}|\,|g_n|$$

so the received power grows as **$N^2$** — the key RIS result. A random-phase RIS only gives a gain proportional to $N$.

### 6. Finite phase resolution

With $b$-bit unit cells the phase step is $\Delta=2\pi/2^b$. The average beamforming power loss factor is

$$\eta(b)=\mathrm{sinc}^2\!\left(\frac{\pi}{2^b}\right)=\left(\frac{\sin(\pi/2^b)}{\pi/2^b}\right)^2$$

| Bits $b$ | Phase levels | Power loss |
| :------: | :----------: | :--------: |
| 1 | 2 | ≈ −3.9 dB |
| 2 | 4 | ≈ −0.9 dB |
| 3 | 8 | ≈ −0.2 dB |

### 7. Performance metrics

**Spectral efficiency and achievable rate**

$$SE=\log_2(1+\gamma)\ \text{[bit/s/Hz]},\qquad R=B\log_2(1+\gamma)\ \text{[bit/s]}$$

**Ergodic capacity**

$$\bar C=\mathbb{E}\big[B\log_2(1+\gamma)\big]$$

**SINR** (when interference is present)

$$\text{SINR}=\frac{S}{I+N}$$

**Outage probability** (for threshold $\gamma_{th}$). For Rayleigh fading, where $\gamma$ is exponentially distributed with mean $\bar\gamma$:

$$P_{out}=\Pr(\gamma<\gamma_{th})=1-\exp\!\left(-\frac{\gamma_{th}}{\bar\gamma}\right)$$

**Bit error rate**

- BPSK over AWGN: $\;BER=Q\!\big(\sqrt{2\gamma}\big)$
- BPSK over Rayleigh fading: $\;BER=\tfrac12\Big(1-\sqrt{\tfrac{\bar\gamma}{1+\bar\gamma}}\Big)$

where $Q(x)=\tfrac{1}{\sqrt{2\pi}}\int_x^\infty e^{-t^2/2}\,dt$.

**Energy efficiency**

$$EE=\frac{R}{P_{total}},\qquad P_{total}=\frac{P_t}{\xi}+P_c+N\,P_{n}(b)$$

with amplifier efficiency $\xi$, circuit power $P_c$ and per-element RIS power $P_n(b)$.

### 8. Beam steering with a RIS

The **generalised reflection law** links the phase gradient across the surface to the reflection angle:

$$\sin\theta_r-\sin\theta_i=\frac{\lambda}{2\pi}\,\frac{d\Phi}{dx}$$

For a uniform linear array with spacing $d$, steering toward $\theta_r$ needs per-element phase

$$\phi_n=-\,\frac{2\pi}{\lambda}\,n\,d\,(\sin\theta_r-\sin\theta_i)$$

---

## 🤖 AI / ML Models

Finding the best phase vector $\boldsymbol\theta$ is a non-convex problem, especially with discrete phases and many elements. AI methods learn the mapping from channel / position information to good RIS configurations.

**Optimisation problem**

$$\max_{\boldsymbol\theta}\;R(\boldsymbol\theta)\quad\text{s.t. }\theta_n\in\left\{0,\tfrac{2\pi}{2^b},\dots,\tfrac{2\pi(2^b-1)}{2^b}\right\}$$

| Approach | Idea | Typical loss / reward |
| -------- | ---- | --------------------- |
| **Supervised DNN** | Input: CSI or user position → Output: phase vector (labels from the co-phasing solution) | MSE or cross-entropy over phase levels |
| **Deep Reinforcement Learning (DQN / DDPG / PPO)** | Agent picks phases, environment returns the rate | Reward $=R$ (sum-rate) or $-$outage |
| **Classical baselines** | Random phase, co-phasing (analytical), alternating optimisation | Used for comparison |

<!-- TODO: state which model(s) the platform actually uses, with its architecture and hyper-parameters. -->

---

## 🏗️ System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                   Dashboard (Frontend)                      │
│   parameter controls · live charts · comparison views       │
└──────────────────────────────┬──────────────────────────────┘
                               │  parameters / results
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│          request handling · orchestration · caching         │
└───────┬──────────────────────┬──────────────────────┬───────┘
        ▼                      ▼                      ▼
┌───────────────┐    ┌───────────────────┐   ┌─────────────────┐
│ Channel & RIS │    │   AI Optimiser    │   │ Metrics Engine  │
│  simulation   │    │ DNN / DRL models  │   │ SNR · rate ·    │
│ path loss,    │───►│ predict phases θ  │──►│ outage · BER ·  │
│ fading, Θ     │    │                   │   │ energy eff.     │
└───────────────┘    └───────────────────┘   └────────┬────────┘
                                                      │
                                                      ▼
                                          Graph data → Dashboard
```

**Data flow:** user parameters → channel simulation → AI predicts RIS phases → metrics computed → results plotted next to theoretical curves.

<!-- TODO: replace with the real modules, framework and data flow of the repository. -->

---

## 📁 Project Structure

```text
AI_6G_RIS_PLATFORM/
│
├── <TODO: entry point, e.g. app / main file>
├── <TODO: channel & RIS simulation module>
├── <TODO: AI model code and trained weights>
├── <TODO: metrics / formulas module>
├── <TODO: dashboard / frontend files>
├── requirements / package file
└── README.md
```

> ⚠️ Replace the placeholders with the real folders and files of the repository.

---

## ⚙️ Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/shubhamk23b/AI_6G_RIS_PLATFORM.git
cd AI_6G_RIS_PLATFORM

# 2. Install dependencies
<TODO: e.g. pip install -r requirements.txt  or  npm install>

# 3. Run the platform
<TODO: e.g. streamlit run app.py  or  npm start>
```

---

## 🎚️ Simulation Parameters

| Parameter | Symbol | Typical range |
| --------- | ------ | ------------- |
| Carrier frequency | $f_c$ | 3.5 GHz – 100 GHz (sub-THz for 6G) |
| Bandwidth | $B$ | 20 MHz – 1 GHz |
| RIS elements | $N$ | 16 – 1024 |
| Phase resolution | $b$ | 1 – 4 bits |
| Transmit power | $P_t$ | 10 – 30 dBm |
| BS–RIS / RIS–user distance | $d_1,d_2$ | 10 – 200 m |
| Rician K-factor | $K$ | 0 – 10 dB |
| Noise figure | $NF$ | 5 – 9 dB |

---

## ⚠️ Assumptions & Limitations

- Idealised unit cells (reflection amplitude $\beta_n=1$ unless stated)
- Perfect or estimated CSI is assumed; real systems need channel estimation, which is costly for large RIS
- Single-user, single-RIS scenario unless extended
- Simulation results are theoretical and not hardware-validated

---

## 🚀 Future Improvements

- [ ] Multi-user and multi-RIS scenarios
- [ ] Imperfect CSI and channel-estimation error modelling
- [ ] Mobility and time-varying channels
- [ ] Hardware impairments and phase-dependent amplitude
- [ ] Real measurement / dataset integration
- [ ] Export of graphs and reports
- [ ] Federated and multi-agent RL for RIS deployment

---

## 📚 References

1. Q. Wu and R. Zhang, "Intelligent Reflecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming," *IEEE Trans. Wireless Commun.*, 2019.
2. E. Basar *et al.*, "Wireless Communications Through Reconfigurable Intelligent Surfaces," *IEEE Access*, 2019.
3. M. Di Renzo *et al.*, "Smart Radio Environments Empowered by Reconfigurable AI Meta-Surfaces," *IEEE JSAC*, 2020.
4. T. S. Rappaport, *Wireless Communications: Principles and Practice*.
5. A. Goldsmith, *Wireless Communications*, Cambridge University Press.

---

## 👨‍💻 Author

**Shubham Kanojiya**
AI/ML & Backend Developer

GitHub: [@shubhamk23b](https://github.com/shubhamk23b)

---

⭐ If you find this project useful, consider giving it a star on GitHub!
