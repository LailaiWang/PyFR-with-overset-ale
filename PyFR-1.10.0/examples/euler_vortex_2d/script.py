import argparse
parser = argparse.ArgumentParser()
parser.add_argument('bmesh', help='bmesh')
parser.add_argument('omesh', help='omesh')
parser.add_argument('nsoln',type=int, help='nsoln')
args = parser.parse_args()

for i in range(args.nsoln):
    a = '{:03}'.format(i)
    cmdrun = f'python3 ~/PyFR-1.10.0/pyfr/__main__.py export {args.bmesh} euler-mul-{a}.pyfrs euler-mul-{a}.vtu -am {args.omesh}'
    
    import os
    os.system(cmdrun)
