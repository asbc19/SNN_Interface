/*
 * Copyright (C) 2016 - 2018 Xilinx, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
 * SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include <stdio.h>
#include <string.h>

#include "lwip/sockets.h"
#include "netif/xadapter.h"
#include "lwipopts.h"
#include "xil_printf.h"
#include "FreeRTOS.h"
#include "task.h"

// New libraries to include
#include "xil_io.h"
#include "xparameters.h"

#define THREAD_STACKSIZE 1024

u16_t echo_port = 7;

void print_echo_app_header()
{
    xil_printf("%20s %6d %s\r\n", "Server 1",
                        echo_port,
                        "$ telnet <board_ip> 7");

}

/* thread spawned for each connection */
void process_echo_request(void *p)
{
	int sd = (int)p;					// Socket
	int RECV_BUF_SIZE = 981;			// Size for receive buffer (Each message is 980 + 1(ID) Bytes)
	int n, nwrote;						// Number of Bytes (receive and send)
	int word_1x8b;						// 1 x 8-bit data (debugging data in BRAM)
	short int i;						// Counter for loops
	short int shift_mem = 0;			// Aim to the memory space for message 1 and 2
	short int shift_t = 323;			// Aim to the memory space for BRAM verification

	char recv_buf[RECV_BUF_SIZE];		// Array of char to receive data from PC through Ethernet
	char aux_buf1[4];					// Obtain the current 4 Bytes from Message to store in BRAM
	char m_id;							// Message ID (1 or 2)
	char send_buf[2];					// Buffer for data to PC
	u32 aux_buf2 = 0;					// uint32_t --> fixed to 4 Bytes of data to BRAM

	while (1) {
		/* read a max of RECV_BUF_SIZE bytes from socket */
		if ((n = read(sd, recv_buf, RECV_BUF_SIZE)) < 0) {
			xil_printf("%s: error reading from socket %d, closing socket\r\n", __FUNCTION__, sd);
			break;
		}

		/* Spikes Data is received */
		if (n > 6) {
			// Define the received message based on the ID (last byte)
			if (recv_buf[980] == 'A') {
				// Message 1 was received
				// We aim for the fist portion of the BRAM
				m_id = '1';
				shift_mem = 0;
				shift_t = 323;

			} else if (recv_buf[980] == 'Z') {
				// Message 2 was received
				// We aim for the second portion of the BRAM
				m_id = '2';
				shift_mem = 980;
				shift_t = 425;

			} else {
				// Wrong message
				xil_printf("No ID in the message!");
				break;
			}

			// Debugging the received message
			xil_printf("\r\nMessage %c: %d Bytes\n", m_id, n-1);

			// Iterate over the whole receiving buffer
			// Each time 4 Bytes are stored in BRAM --> repeat 245 times (980 / 4)
			for (i = 0; i < 245; i++) {
				// Retrieve 4 Bytes from receiving buffer
				aux_buf1[0] = recv_buf[4*i];
				aux_buf1[1] = recv_buf[4*i + 1];
				aux_buf1[2] = recv_buf[4*i + 2];
				aux_buf1[3] = recv_buf[4*i + 3];

				// Copy from aux. buffer 1 to 2
				// to keep the order (FIFO) and int is required
				// (dest, scr, #of Bytes)
				memcpy(&aux_buf2, aux_buf1, sizeof(aux_buf2));

				// Write data in BRAM
				Xil_Out32(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + 4*i + shift_mem, aux_buf2);
			}

			// Verify data in BRAM
			// 1x8-bit data (1 bytes)

			// First and Last Byte in each message
			word_1x8b = Xil_In8(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + 0 + shift_mem);
			xil_printf("\r1st Byte from Message %c = 0x%02x", m_id, word_1x8b);
			xil_printf("\rBRAM Address: 0x%08x", XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + 0 + shift_mem);

			word_1x8b = Xil_In8(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + 979 + shift_mem);
			xil_printf("\rLast Byte from Message %c = 0x%02x", m_id, word_1x8b);
			xil_printf("\rBRAM Address: 0x%08x", XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + 979 + shift_mem);

			// Time steps: t=3 and t=14
			word_1x8b = Xil_In8(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + shift_t + shift_mem);
			xil_printf("\r1st Verification Byte from Message %c = 0x%02x",m_id, word_1x8b);
			xil_printf("\rBRAM Address: 0x%08x", XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + shift_t + shift_mem);

			word_1x8b = Xil_In8(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + shift_t + 1 + shift_mem);
			xil_printf("\r2nd Verification Byte from Message %c = 0x%02x",m_id, word_1x8b);
			xil_printf("\rBRAM Address: 0x%08x", XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + shift_t + 1 + shift_mem);

			word_1x8b = Xil_In8(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + shift_t + 2 + shift_mem);
			xil_printf("\r3rd Verification Byte from Message %c = 0x%02x",m_id, word_1x8b);
			xil_printf("\rBRAM Address: 0x%08x", XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + shift_t + 2 + shift_mem);

			word_1x8b = Xil_In8(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + shift_t + 3 + shift_mem);
			xil_printf("\r4th Verification Byte from Message %c = 0x%02x",m_id, word_1x8b);
			xil_printf("\rBRAM Address: 0x%08x", XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + shift_t + 3 + shift_mem);

			word_1x8b = Xil_In8(XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + shift_t + 4 + shift_mem);
			xil_printf("\r5th Verification Byte from Message %c = 0x%02x",m_id, word_1x8b);
			xil_printf("\rBRAM Address: 0x%08x", XPAR_AXI_BRAM_CTRL_0_S_AXI_BASEADDR + shift_t + 4 + shift_mem);

			// Simulate the result from the SNN chip
			// 1st Byte has the classification result
			// 2nd Byte is the ID = 0x52
			send_buf[0] = 0x07;
			send_buf[1] = 'R';
		}

		/* Verify the initial connection if message = "verify" */
		if (!strncmp(recv_buf, "verify", 6)) {
			xil_printf("\n\rClient started connection");

			// Prepare the confirmation
			// 1st Byte is the confirmation: C = 0x43
			// 2nd Byte is the ID = 0x52
			send_buf[0] = 'C';
			send_buf[1] = 'R';
			// Send the confirmation
			write(sd, send_buf, 2);
		}

		/* Break if the received message = "quit" */
		if (!strncmp(recv_buf, "quit", 4)) {
			xil_printf("\n\rClient end connection");
			break;
		}

		/* Break if client closed connection */
		if (n <= 0)
			break;

		/* Handle request */
		// Send the result from SNN chip in 2 Bytes
		// It needs synchronization with the result from SNN chip
		// But for the moment, we will answer after the message 2
		if (m_id == '2') {
			if ((nwrote = write(sd, send_buf, 2)) < 0) {
				xil_printf("%s: ERROR responding to client echo request. received = %d, written = %d\r\n",
						__FUNCTION__, n, nwrote);
				xil_printf("Closing socket %d\r\n", sd);
				break;
			} else {
				xil_printf("\n\r%d Bytes sent to PC", nwrote);
				xil_printf("\rSent result from SNN chip: 0x%02x", send_buf[0]);
			}
		}
	}

	/* close connection */
	close(sd);
	vTaskDelete(NULL);
}

void echo_application_thread()
{
	int sock, new_sd;
	int size;
#if LWIP_IPV6==0
	struct sockaddr_in address, remote;

	memset(&address, 0, sizeof(address));

	if ((sock = lwip_socket(AF_INET, SOCK_STREAM, 0)) < 0)
		return;

	address.sin_family = AF_INET;
	address.sin_port = htons(echo_port);
	address.sin_addr.s_addr = INADDR_ANY;
#else
	struct sockaddr_in6 address, remote;

	memset(&address, 0, sizeof(address));

	address.sin6_len = sizeof(address);
	address.sin6_family = AF_INET6;
	address.sin6_port = htons(echo_port);

	memset(&(address.sin6_addr), 0, sizeof(address.sin6_addr));

	if ((sock = lwip_socket(AF_INET6, SOCK_STREAM, 0)) < 0)
		return;
#endif

	if (lwip_bind(sock, (struct sockaddr *)&address, sizeof (address)) < 0)
		return;

	lwip_listen(sock, 0);

	size = sizeof(remote);

	while (1) {
		if ((new_sd = lwip_accept(sock, (struct sockaddr *)&remote, (socklen_t *)&size)) > 0) {
			sys_thread_new("echos", process_echo_request,
				(void*)new_sd,
				THREAD_STACKSIZE,
				DEFAULT_THREAD_PRIO);
		}
	}
}
