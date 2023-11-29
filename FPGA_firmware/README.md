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
