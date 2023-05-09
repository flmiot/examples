import os

base = r'/home/otteflor/scratch/tmp'
target = r'/home/otteflor/Git/euxfel_reduction/config'

folders = sorted(os.listdir(base))
for folder in folders:
    p = os.path.join(base, folder)
    
    files = os.listdir(p)
    
    print(files)
    
    if '09_assemble.h5' in files:
        with open(os.path.join(p, '09_assemble.h5'), 'rb') as source_file:
            b = source_file.read()
            
        fp = os.path.join(target, folder, '09_assemble.h5')
        with open(fp, 'wb') as target_file:
            target_file.write(b)