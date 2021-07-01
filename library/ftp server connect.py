from ftplib import FTP

ftp = FTP('ftp.ihostfull.com')   # connect to host, default port

ftp.login('uoolo_22913024','Mn&menthuhostm0rty')               # user anonymous, passwd anonymous@

ftp.cwd(r'/lostpandas.co.uk/htdocs/th data')

ftp.retrlines('LIST') 

print(ftp.pwd())

localfile = r"C:\Users\Phil\Desktop\th.txt"
remotefile = 'test.txt'

with open(localfile, "rb") as f:
	ftp.storbinary('STOR %s' % remotefile, f)

ftp.retrlines('LIST') 

ftp.quit()

