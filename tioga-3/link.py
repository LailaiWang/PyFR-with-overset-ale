import os
path= '../../lib/python3.9/site-packages'

cmd = 'rm ../../lib/python3.9/site-packages/convert.py ../../lib/python3.9/site-packages/_convert.so ../../lib/python3.9/site-packages/tioga.py ../../lib/python3.9/site-packages/_tioga.so' 
os.system(cmd)

cmd = f'cp bin/convert.py {path}/convert.py'
os.system(cmd)

cmd = f'cp bin/_convert.so {path}/_convert.so'
os.system(cmd)
cmd = f'cp  bin/_tioga.so {path}/_tioga.so'
os.system(cmd)
cmd = f'cp bin/tioga.py {path}/tioga.py'
os.system(cmd) 
 
