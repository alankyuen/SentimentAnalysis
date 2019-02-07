#

import subprocess

for i in range(10):
url = "http://localhost:900{}/shutdown?".format(i)
shutdown_key = subprocess.getoutput("cat /tmp/corenlp.shutdown")
r = requests.post(url,data="",params={"key": shutdown_key})

print("hello")