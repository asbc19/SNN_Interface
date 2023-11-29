
import telnetlib
  
# Connect to the Telnet server
tn = telnetlib.Telnet("192.168.1.10", 7)
loop = True

while(loop):
    
    # Wait for the command
    k_command = input("Enter your command: ")

    if k_command == 'exit':
        loop = False
    # Send a command to the server
    # We need to send it as bytes
    message_byte = str.encode(k_command)
    tn.write(message_byte)   

    # Read the output from the server
    output = tn.read_some()     # Read the previous message
    #output = tn.read_eager()   # Miss to read the first time
    #output = tn.read_all()     # read until the EOF
    
    #Debug
    #print("Pass the reading")

    # Print the output
    print(f'FPGA answer: {output}')
    #tn.close() 
  
tn.close()