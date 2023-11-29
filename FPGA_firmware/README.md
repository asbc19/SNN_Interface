## Files
The following files are included:
- FPGA bitstream.
- Modified FreeRTOS lwIp Echo Server Example.
  - Modified main.c.
  - Change Echo.c for server1.c.
- Test example for telnet in Python.

## Notes
1. There is an error when creating FreeRTOS lwIp Echo Server Example. To solve it, change to `lwip/timeouts.h` in `xemacliteif.c`
![Fix_sdk](https://github.com/asbc19/SNN_Interface/assets/67765415/cca58932-f1c7-43cc-8be0-135692d8f846)

2. Load the bitstream.
https://github.com/asbc19/SNN_Interface/assets/67765415/10d9e5fa-ca47-4046-8ade-244a32ded778

3. To run the server, connect the UART port with a baud rate of 9600 and then run the firmware.
https://github.com/asbc19/SNN_Interface/assets/67765415/25c134c3-2987-45a7-9312-e1c55759dd08
