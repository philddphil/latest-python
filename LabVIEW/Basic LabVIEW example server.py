##############################################################################
# Import some libraries
##############################################################################

import time
import socket

##############################################################################
# Do some stuff
##############################################################################

# This server needs to be run in the background, in conjunction with a labVIEW
# client. These should be in some folder nearby.

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 8089))
server.listen(1)
print('Server Listening')

while True:
    conn, addr = server.accept()
    # Assigns the received data (bytes) from LabVIEW to 'cmnd'
    cmnd = conn.recv(16384)
    call_time = time.ctime()

    if 'INIT' in str(cmnd):
        # Super well commented code describes what is going on 
        # during initialisation
        print(call_time + ' INIT')
        conn.sendall(b'INIT-DONE_')

    elif 'ACTION1' in str(cmnd):
        # Generate a hologram and save it as a bmp
        print('ACTION1')
        conn.sendall(b'ACTION-DONE')
        
    elif 'QUIT' in str(cmnd):
        # Closes the server program
        print(call_time + ' QUIT')
        conn.sendall(b'QUIT-DONE_')
        break

server.close()
