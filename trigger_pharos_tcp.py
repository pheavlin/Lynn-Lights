import socket

s = socket.socket()
s.connect(('169.254.183.163',22000))
s.send("a\r".encode())
s.close()