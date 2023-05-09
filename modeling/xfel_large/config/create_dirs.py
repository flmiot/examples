import os
import yaml

base = r'2699_094_on'

scratch_base = r'/home/otteflor/scratch/tmp_fit'

base_files = os.listdir(base)

for i in [81,82, 94, 95]:
    folder_name = r'2699_{:0>3d}_off'.format(i)

    try:
        os.mkdir(folder_name)
    except Exception as e:
        print(e)

    for f in base_files:
        try:
            with open(os.path.join(base, f), 'rb') as file:
                b = file.read()

            with open(os.path.join(folder_name, f), 'wb') as file:
                file.write(b)
        except IsADirectoryError as e:
            print(e)


    with open(os.path.join(base, '02_choose.yml'), 'r') as file:
        config = yaml.load(file.read(), Loader = yaml.SafeLoader)
    config['fit_data']['file'] = r'/home/otteflor/scratch/tmp/2699_{:0>3d}/09_assemble.h5'.format(i)
    with open(os.path.join(folder_name, '02_choose.yml'), 'w+') as file:
        yaml.dump(config, file, default_flow_style=False)


    with open(os.path.join(base, 'Makefile'), 'r') as file:
        l = list(file.readlines())

    l[2] = 'SCRATCH = /home/$(USER)/scratch/tmp_fit/{}\n'.format(folder_name)

    with open(os.path.join(folder_name, 'Makefile'), 'w+') as file:
        file.write("".join(l))
        
        
    try:
        os.mkdir(os.path.join(scratch_base, folder_name))
    except Exception as e:
        print(e)
