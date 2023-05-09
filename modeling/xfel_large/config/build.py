import os
import subprocess
import multiprocessing

user = os.environ.get('USERNAME')

runs = ['4a', '4b', '4bh2o', '6a', '6b', '12h2o', '13', '14', '16']

def make(directory):
    subprocess.call(["make"], cwd=directory)

if __name__ == '__main__':
    directories = [r'C:/Users/{}/Git/euxfel_fit/config/{}'.format(user, r) for r in runs]
    print(directories)
    with multiprocessing.Pool(processes = 1) as p:
        p.map(make, directories)
