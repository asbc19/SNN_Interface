# Interface for SNN chip
This repository includes the source code to generate, transmit, and store spike trains for a Spiking Neural Network (SNN) chip.  
- It additionally provides the code to train and quantize weights for an SNN model.
- It uses the MNIST dataset.

## 1. PC-FPGA Interface
The basic operation of these source codes are:
1. After selecting a sample, the image is codified into spikes (rate encoding).
2. The codified image is sent to the FPGA (server) using Ethernet (Telnet protocol).
3. The FPGA stores the received information in a BRAM for later use.

### 1.1 PC GUI 
The GUI uses [PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI.git) and [BindsNET](https://github.com/BindsNET/bindsnet).  

To run the code, create the following conda environment:
```
conda create --name snn_env python==3.8.10

conda install -c conda-forge pysimplegui

conda install git
pip install git+https://github.com/BindsNET/bindsnet.git
```
https://github.com/asbc19/SNN_Interface/assets/67765415/22383ff6-ff38-480c-8045-ea15bd75e4ff

### 1.2. FPGA Implementation
Download the complete [Vivado Project](https://1drv.ms/f/s!Ar7U4hJqERkwgoU5jZTItalcBQ_r3Q?e=I4ahLl). 
- This implementation was created in Vivado 2018.3.
- The target device is Kintex-7 KC705 Evaluation Platform.
- It implements uBlaze + UART + Ethernet Blocks.
- The firmware for uBlaze was created using SDK 2018.3.  
