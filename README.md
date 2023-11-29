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

### 1.2 FPGA Bitstream and Firmware
Download the complete [Vivado Project](https://1drv.ms/f/s!Ar7U4hJqERkwgoU5jZTItalcBQ_r3Q?e=I4ahLl). 
- This implementation was created in Vivado 2018.3.
- The target device is Kintex-7 KC705 Evaluation Platform.
- It implements uBlaze + UART + Ethernet Blocks.
- The firmware for uBlaze was created using SDK 2018.3.
- To run the server, connect the UART port with a baud rate of 9600 and load the firmware.

https://github.com/asbc19/SNN_Interface/assets/67765415/3c8997e0-a3ab-4190-a9b0-39a3b73e7e5b

### 1.3 PC-FPGA Communication
Make sure that the PC is in the same network as the FPGA server.
- Set IP in the PC to 192.168.8.
  
https://github.com/asbc19/SNN_Interface/assets/67765415/0928d5e9-6e01-4f29-8c9e-2ee888f1b989
