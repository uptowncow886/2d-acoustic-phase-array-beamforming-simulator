# 2D Acoustic Phased Array Beamforming Simulator

Interactive Python simulator for a **phased array receiver**.  
Visualizes beam patterns with steering angle control, tilting, and both 2D and 3D plots.

![3D Beam Pattern](simulation_result/sim_result_3D_+_7elements.png)
![3D Line Plot](simulation_result/sim_result_3D_line_9elements.png)
![Tilting Animation](simulation_result/sim_tilting.mp4)

## Features
- Configurable NxN element phased array (tested with 9x9)
- Programmable frequency and element spacing
- Beam steering and mechanical/electrical tilting visualization
- 2D heatmap and 3D beam pattern rendering
- Real-time parameter adjustment via GUI

## Installation

```bash
git clone https://github.com/uptowncow886/2d-acoustic-phase-array-beamforming-simulator.git
cd 2d-acoustic-phase-array-beamforming-simulator
pip install -r requirements.txt
