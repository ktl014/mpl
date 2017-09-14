Author: Kevin Le
Date: Thursday, September 14, 2017 10:38:00

This document explains the requirements needed to access the scripts and images for the MPL Summer 2017 experiments.

# Connecting to SVCL GPU Server
Two things are needed:
1. Cisco AnyConnect VPN client
- Download it from here: https://blink.ucsd.edu/technology/network/connections/off-campus/VPN/index.html
2. An SSH Client (Putty, Mac Terminal, etc)

Once both are installed, first use the VPN client to connect to the lab. Follow instructions below to do so:
- Be sure to edit security settings to allow connection, before connecting. For instance, uncheck the option "Block connections to untrusted servers"
1. Connect to "vpn.svcl.ucsd.edu"
2. Group: "RA"
3. Input Username/Password
- Contact me @ kevin.le@gmail.com for account information

Launch SSH Client with the following information
IP Address: 192.168.65.21 (gpu1)
Port Number: 22
Username and password are the same as the username and password for connecting to the lab through the VPN client

Finally, after logging into the lab server, enter the command line below to redirect yourself to the main directory for the experiments
>> cd /data4/plankton_wi17/mpl

# Alternative Option
Alternative options include using TeamViewer to directly access the lab computers and using Putty to login into the server
- Download it from here for your respective operating system: https://www.teamviewer.com/en/download/mac/
- To access teamviewer, contact me @ kevin.le@gmail.com for the account information




