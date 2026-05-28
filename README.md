
V0.9 by Fu-Hwei Hong 

This is a simple demonstation simulation to visulize the phase_array receiver beam pattern.

User can program the 9x9 elements of receiver
User can program the receiver element spacing
User can program the input audio wave frequency

phase-array-beamforming-simulator/
├── README.md                  ← This file
├── LICENSE                    ← (MIT)
├── requirements.txt           ← List of Python packages
├── app-v3.py                  ← Main script
├── src/                       ← 
├── examples/                  ← 
├── results/                   ← Screenshots, GIFs, or sample plots
├── docs/                      ← Extra documentation (optional)
└── .gitignore                 ← Ignore cache files, etc.


The code is done with google antigravity then chatgpt coding 
# 2D Acoustic Phased Array Beamforming Simulator

Interactive Python simulator for a **9x9 programmable phased array** receiver.  
Visualizes beam patterns with steering angle control and 2D/3D plots.

![Beam Pattern Example](simulation_result/beam_pattern.png)   <!-- Add your image here -->

## Features
- 9x9 element phased array simulation
- Programmable element spacing and frequency
- Beam steering and tilting visualization
- 2D heatmap + 3D beam pattern plots
- Real-time parameter adjustment

## Installation

```bash
git clone https://github.com/uptowncow886/2d-acoustic-phase-array-beamforming-simulator.git
cd 2d-acoustic-phase-array-beamforming-simulator
pip install -r requirements.txt
