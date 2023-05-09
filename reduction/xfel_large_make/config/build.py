import os
import subprocess
import multiprocessing

user = os.environ.get('USER')

# runs = [22, 23, 24, 26, 27, 28, 29, 30, 34, 36, 37, 38, 39, 43, 44, 45, 53, 54, 55, 56, 57, 58, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104]
# runs = [73,74,75,81,82,93, 94,95,102]
# runs = [64, 65, 67, 68, 69, 70, 71, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92]

runs = [1]

def make(directory):
    subprocess.call(["make"], cwd=directory)

if __name__ == '__main__':
    directories = ['/home/{}/Git/euxfel_reduction/config/2699_{:0>3d}'.format(user, r) for r in runs]
    with multiprocessing.Pool(processes = 64) as p:
        p.map(make, directories)