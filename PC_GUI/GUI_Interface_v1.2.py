"""
Date: 2023 - 08 - 17
Author: Andres Brito.

FGPA-PC GUI Interface v1.2
- Random selection of samples numbers (images)
- Show 4 options
- An specific image coded on spikes using Poisson distribution.
    * Each pixel has a train of spikes with an specific length.
- The total number of bytes are sent to the FPGA (server) using Ethernet (Telnet protocol).
- Message from PC is divided in two: 2*981 Bytes
- Each message has an ID in the last byte:
    - Message 1 ID: A (0x41)
    - Message 2 ID: Z (0x5A)
- FPGA answer also has an ID:
    - FPGA message ID: R (0x52)
- Any message from the FPGA can be received to illustrate birectional communication.
- Include the Raster Plot for all spikes
"""

###################################################
# Libraries
###################################################
# For GUI
import PySimpleGUI as sg
import os

# For Spike Generation
import torch
import numpy as np
from torchvision import datasets, transforms
from bindsnet.encoding import poisson

# Use for Ethernet Communication
import telnetlib

# To use plots in canvas
from matplotlib import use as use_agg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Use Tkinter Agg
use_agg('TkAgg')

###################################################
# Functions
###################################################

# Prepare the MNIST dataset
"""
IN: 
v_show --> show information regarding the test image 

OUT:
mnist_loader --> iterable with individual images and corresponding labels
"""
def prepare_dataset(v_show=False):
    # Download and prepare MNIST dataset
    # The dataset is stored in the current directory
    mnist_dataset = datasets.MNIST(root=my_path + "\Dataset_MNIST", train=False, download=True, 
                                transform=transforms.Compose([transforms.ToTensor()]))

    # Contruct an iterable with individual images (batch size = 1)
    mnist_loader = torch.utils.data.DataLoader(dataset=mnist_dataset, batch_size=100, shuffle=False)

    # Show information of test sample
    if v_show:
        # Get the next element (the 1st image)
        images, labels = next(iter(mnist_loader))   
        print("Dataset is ready")            
        print(f'\rIndividual Image Size: {images[0].shape}\n')

    # Return the dataloader
    return mnist_loader

# Encode images to spikes
# each pixel has a spikes train
# Visualize the spikes train for a specific pixel (optional)
# For the moment, we only take one sample image (1st image)
"""
IN: 
data --> image to encode
intensity --> intensity of each pixel (SNN paper). Default value 256 (Prof. Wang)
time_spike --> length of Poisson spike train in units of time. Default value 20 (Prof. Wang)
v_show --> show information

OUT:
spike_in --> Each pixel has a spike train of duration time_spike
    for MNIST: 28 x 28 spike trains of length 250
"""
def encode_image(data, intensity=256, time_spike=20, v_show=False):
    ###################################################
    # MNIST image are converted to Poisson-spike 
    # with firing rates proportional to the intensity 
    # of the corresponding pixel.	
    ###################################################
    
    # Change each pixel to a proportion of the intensity
    data = data * intensity
    #print(f'Intensity of Image data: {data}')

    # Generate spikes using Poisson distribution
    # the lenght of the spike is time_spike [units of time]
    spikes = poisson(data, time=time_spike)

    # Generate input data
    # Each pixel (28 x 28 = 784) has its spike at each time step
    # We store each spike train the corresponding column
    spike_in = np.reshape(spikes.numpy(), (time_spike, 784))
    
    if v_show:
        # Data information
        print(f'\nLength of Spike: {time_spike} [time step]')
        print(f'\rShape of data: {data.shape}')

        # Result information
        print(f'\rOriginal Spikes shape: {spikes.shape}')
        print(f'\rInput Spikes: {spike_in}')
        print(f'\rInput Spikes shape: {spike_in.shape}')

        # Search for the spikes to plot
        #pos_spike = np.nonzero(spike_in)
        #print(pos_spike)

        # Show some spikes generated
        # Create a generic figure
        fig, ax = plt.subplots(figsize=(5, 4))
        # Clear the current axis
        ax.cla()
        ax.plot(spike_in[3,:])
        ax.set_title("Spikes for t=3 (time step 4)")
        # Save the image
        fig.savefig(my_path + "\Results\spike_t3.png")

        # # Show some spikes generated
        # Create a generic figure
        fig, ax = plt.subplots(figsize=(5, 4))
        # Clear the current axis
        ax.cla()
        ax.plot(spike_in[14,:])
        ax.set_title("Spikes for t=14 (time step 15)")
        # Save the image
        fig.savefig(my_path + "\Results\spike_t14.png")

        # Raster plot for all the neurons:
        # Vector with all the time steps:
        time_v = np.arange(1, time_spike+1, 1)
        # Time steps per neuron
        neuronD = time_v*np.transpose(spike_in)
        # Create a generic figure
        fig, ax = plt.subplots(figsize=(8,4))
        # Clear the current axis
        ax.cla()
        ax.eventplot(neuronD, linelengths=1)
        ax.set_title("Spike Raster Plot")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron")
        ax.set_xlim(1,20)
        ax.set_ylim(0,784)
        # Save the image
        fig.savefig(my_path + "\Results\plotRaster.png")

    return spike_in

# Create the message to send through Ethernet
# Turn array of zeros and ones to string of bytes
"""
IN: 
spike_in --> Each pixel has a spike train of duration time_spike
n_pixel --> Total number of pixels in the image (define the number of bytes).
v_show --> show information

OUT:
spike_m0 --> spike trains in a string organized by time step: t [0,9]
spike_m1 --> spike trains in a string organized by time step: t [10,19]
"""
def build_message(spike_in, n_pixel, v_show=False):
    
    # Get the number of time steps
    n_r, _ = spike_in.shape

    # Iterate in each row to get the equivalent bytes for each time step (all pixels)
    for i in range(n_r):
        
        # Create a string for the correponding time step
        str_train = "".join(spike_in[i,:].astype(str))

        # Turn each binary string to integer
        int_train = int(str_train, 2)
        
        # Turn each integer into correponding byte string
        # Number of bytes corresponds to the number of pixels
        byte_train = int_train.to_bytes(int(n_pixel/8), byteorder='big')

        # Only accumulate from the second time step
        if i==0:
            spike_message = byte_train
        else:
            spike_message += byte_train

        # Store t = 3 (time step 4) for verification
        if i == 3 and v_show:
            # Get the bytes
            t3_b = byte_train
        
        # Store t = 14 (time step 15) for verification
        if i == 14 and v_show:
            # Get the bytes
            t14_b = byte_train
    
    # Add ID to each message
    # Message 1 ID: A (0x41)
    # Message 2 ID: Z (0x5A)
    spike_m0 = spike_message[0:980] + bytes("A", 'ascii')
    spike_m1 = spike_message[980:] + bytes("Z", 'ascii')

    # Information
    if v_show:
        print("\nMessage 1 (Bytes):")
        print(spike_m0)    
        print(f'Number of Bytes: {len(spike_m0)} Bytes')
        print(f'Message ID: {hex(spike_m0[-1])}')

        print("\nMessage 2 (Bytes):")
        print(spike_m1)    
        print(f'Number of Bytes: {len(spike_m1)} Bytes')
        print(f'Message ID: {hex(spike_m1[-1])}')
        
        print(f'\nTotal number of Bytes (20 x {int(n_pixel/8)} + 2): {len(spike_m0) + len(spike_m1)} Bytes')

        # Debug: Hex and Bytes corresponding to time step t = 3 (Message 1)
        print(f'\nt=3 (time step 4) in Hex: 0x{t3_b.hex()}')
        print(f't=3 (time step 4 ) in Bytes): {t3_b}')

        # We will use 5 bytes to verify the message 1 (Byte[30:34]) 
        print(f'5 Bytes to verify the message 1: Byte 30-34 in t=3: 0x{t3_b.hex()[29*2:34*2]}')

        # Debug: Hex and Bytes corresponding to time step t = 14 (Message 2)
        print(f'\nt=14 (time step 15) in Hex: 0x{t14_b.hex()}')
        print(f't=14 (time step 15 ) in Bytes): {t14_b}')

        # We will use 5 bytes to verify the message 2 (Byte[34:38])
        print(f'5 Bytes to verify the message 2: Byte 34-38 in t=14: 0x{t14_b.hex()[33*2:38*2]}')

        # Extract the portion of t=3 from the message
        print('\n\n#################################################################')
        print('## For Verification:')
        print('#################################################################')

        m1 = spike_message[0:980]
        t3_message = m1[98*3 + 30 - 1:98*3 + 34]
        print(f'\nByte 30-34 from t=3 in message 1 (Bytes): {t3_message}')
        print(f'Byte 30-34 from t=3 in message 1 (Hex): 0x{t3_message.hex()}\n')

        # Extract the portion of t=14 from the message
        m2 = spike_message[980:]
        t14_message = m2[98*(14-10) + 34 - 1:98*(14-10) + 38]
        print(f'Byte 34-38 from t=14 in message 2 (Bytes): {t14_message}')
        print(f'Byte 34-38 from t=14 in message 2 (Hex): 0x{t14_message.hex()}\n')
    
    return spike_m0, spike_m1 

###################################################
# GUI Layout
###################################################

# 1st Column: Numbers, Codify, and Refresh buttons
column_1 = [
    [
        # 1st Row
        sg.Text("Option 1:"),
        sg.Image(key='-N0-'),
    ],

    [
        # 2nd Row
        sg.Text("Option 2:"),
        sg.Image(key='-N1-'),
    ],

    [
        # 3rd Row
        sg.Text("Option 3:"),
        sg.Image(key='-N2-'),
    ],

    [
        # 4th Row
        sg.Text("Option 4:"),
        sg.Image(key='-N3-'),
    ],

    [
        # 5th Row
        sg.Combo(['1','2','3','4'], default_value='1', key="-Nselect-"),
        sg.Button("Codify"),
        sg.Button("Refresh"),
    ],

]

# 2nd Column: Spikes, Result, exit and Send buttons.
column_2 = [
    [
        # 1st Row
        sg.Text("Label Selected Image:"),
        sg.Text("", size=(0,1), key="-Img_l-"),
    ],

    [
        # 2nd Row
        sg.Image(key='-Spikes_g-'),
    ],

    [
        # 3rd Row
        sg.Text("FPGA Result:"),
        sg.Text("", size=(0,1), key="-FPGA_r-"),
    ],

    [
        #4th Row
        sg.Text("Server Status:"),
        sg.Text("Disconnected", size=(0,1), key="-FPGA_s-"),
        sg.Button("Connect"),
        sg.Button("Disconnect"),

    ],

    [
        # 5th Row
        sg.Button("Send"),
    ],

    [
        # 6th Row
        sg.Button("Exit"),
    ],

]

# Full Layout
layout = [
    [
        sg.Column(column_1),
        # sg.VSeparator(),
        sg.Column(column_2),
    ]
]

###################################################
# Window and Global Elements
###################################################
# Path to the current directory
my_path = os.path.dirname(os.path.abspath(__file__))

# # To allow reproducibility
seed = 110
torch.manual_seed(seed)

# Prepare dataset
mnist_loader = prepare_dataset(v_show=False)

# Obtain 100 samples (image, label) (1 batch)
images, labels = next(iter(mnist_loader))  
# Counter to iterate in the previous array
cnt_img = 96

# SNN parameters: Intensity and Spike Train length
intensity = 256
# Each time step is 98 Bytes (784/8 = 98)
time_spike = 20         # Number of time steps (bits per spike train)

# Create the Window
window = sg.Window("FPGA-PC Interface", layout)

# Turn-off the interative mode
plt.ioff()

###################################################
# Event Loop
###################################################
while True:
    event, values = window.read()

    # Connect to server
    if event == "Connect":
        # Connect to the Telnet server (FPGA server)
        tn = telnetlib.Telnet("192.168.1.10", 7)

        # Send the connect confirmation
        tn.write(bytes("verify", 'ascii'))

        # Read the answer from the server
        fpga_verify = tn.read_some()     # Read the previous message
        #fpga_message2 = tn.read_some()     # Read the previous message
        #output = tn.read_eager()   # Miss to read the first time
        #output = tn.read_all()     # read until the EOF

        # Verify the answer (C = 0x43 = 67 and R =  0x52 = 82)
        if fpga_verify[1] == 82 and fpga_verify[0] == 67:
            # Server answer with verification
            print(f'FPGA verification answer: {fpga_verify[0]}')
            # Update the server status
            window['-FPGA_s-'].update(value="Connected")

        else:
            print("Error in connecting to Server")

    # Disconnect from the server
    if event == "Disconnect":
        # Send the command to disconnect from the server
        tn.write(bytes("quit", 'ascii'))
        # Close the connection
        tn.close()

        # Update the server status
        window['-FPGA_s-'].update(value="Disconnected")        

    # Show samples from the MNIST Dataset
    if event == "Refresh":
        
        # Check the counter (Lower bound of the interval)
        if cnt_img == 96:
            # Re-start the counter
            cnt_img = 0
        else:
            cnt_img += 4
        # print(cnt_img)

        # Iterate in the array images: 
        # select 4 images to display
        for i in range(4):
            # Create a generic figure
            fig = plt.figure(figsize=(2, 2), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])  # span the whole figure
            # Clear the current axis
            ax.cla()
            ax.imshow(images[i+cnt_img].reshape(28,28), cmap="gray")
            ax.set_axis_off()
        
            # Save the image
            fig.savefig(my_path + "\Samples\sample_" + str(i) + ".png")

            # Select the corresponding image
            image_ID = '-N' + str(i) + '-'

            window[image_ID].update(filename = my_path + "\Samples\sample_" + str(i) + ".png")

    # Codify the selected image
    # Show the raster plot
    if event == "Codify":
        # Get the current value in the combo element
        # Turn into index (i)
        opt_ID = int(window['-Nselect-'].get()) - 1
        
        # Retrieve the correponding selected image and label
        sel_img = images[opt_ID + cnt_img]
        sel_lab = labels[opt_ID + cnt_img]
        sel_lab = sel_lab.numpy()
        # print(sel_lab)

        # Show the label of the selected image
        window['-Img_l-'].update(value=sel_lab)

        # encode the selected image
        spike_in = encode_image(data=sel_img, intensity=intensity, 
                                time_spike=time_spike, v_show=True)
        
        # Show the Raster Plot
        window['-Spikes_g-'].update(filename =my_path + "\Results\plotRaster.png")

    # Build the message and
    # send it using Ethernet
    if event == "Send":
        # Generate the messages
        # Each image is codified in two messages
        spike_m0, spike_m1 = build_message(spike_in, n_pixel=784,
                                    v_show=True)
        print(f'Messages were generated!\n')  

        # Send message 1 to server
        tn.write(spike_m0)

        # Send message 2 to server
        tn.write(spike_m1)

        # Read the output from the server
        fpga_message1 = tn.read_some()     # Read the previous message
        #fpga_message2 = tn.read_some()     # Read the previous message
        #output = tn.read_eager()   # Miss to read the first time
        #output = tn.read_all()     # read until the EOF

        # Verify the answer (R =  0x52 = 82)
        if fpga_message1[1] == 82:
            # Print the output
            print(f'FPGA answer: {fpga_message1[0]}')
        else:
            print("Error in message from FPGA")

        # Update the FPGA result
        window['-FPGA_r-'].update(value=fpga_message1[0])

    # Close the window
    if event == "Exit" or event == sg.WIN_CLOSED:
        # Send the command to stop disconnect from the server
        # tn.write(bytes("quit", 'ascii'))
        break

# Close the connection
# tn.close()
# Close the window
window.close()