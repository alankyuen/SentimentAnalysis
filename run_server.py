#

import subprocess
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-port', type=str, default = "9000", help="")
port_num = parser.parse_args().port
wd = os.getcwd()
print(wd)


os.chdir("StanfordModels/stanford-corenlp-full-2018-02-27/")
command = 'java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port {} -timeout 100000  --add-modules java.se.ee'.format(port_num)
server = subprocess.Popen(command, stdout=subprocess.PIPE, shell = True)
print(server.stdout.read())
os.chdir(wd)

# import time
# time.sleep(100)
# import requests
# from subprocess import getoutput

# url = "http://localhost:{}/shutdown?".format(port_num)
# shutdown_key = getoutput("cat /tmp/corenlp.shutdown")
# r = requests.post(url,data="",params={"key": shutdown_key})

# print("hello")