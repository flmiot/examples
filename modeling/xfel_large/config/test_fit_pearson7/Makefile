# Default fitting makefile
SCRIPTS = /home/$(USER)/Git/euxfel_fit
SCRATCH = /home/$(USER)/scratch/tmp_fit/test_fit_pearson7

all : $(SCRATCH)/01_generate.h5 $(SCRATCH)/02_choose.h5 $(SCRATCH)/03_optimize.h5 $(SCRATCH)/04_compare.h5

$(SCRATCH)/01_generate.h5 : 01_generate.yml $(SCRIPTS)/01_generate.py
	python $(SCRIPTS)/01_generate.py $(SCRATCH) ./
    
$(SCRATCH)/02_choose.h5 : $(SCRATCH)/01_generate.h5 01_generate.yml 02_choose.yml $(SCRIPTS)/02_choose.py
	python $(SCRIPTS)/02_choose.py $(SCRATCH) ./
    
$(SCRATCH)/03_optimize.h5 : $(SCRATCH)/02_choose.h5  01_generate.yml 03_optimize.yml $(SCRIPTS)/03_optimize.py
	python $(SCRIPTS)/03_optimize.py $(SCRATCH) ./

$(SCRATCH)/04_compare.h5 : $(SCRATCH)/03_optimize.h5 $(SCRATCH)/01_generate.h5 01_generate.yml 04_compare.yml $(SCRIPTS)/04_compare.py
	python $(SCRIPTS)/04_compare.py $(SCRATCH) ./