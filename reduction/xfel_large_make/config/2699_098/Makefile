# Default reduction makefile
SCRIPTS = /home/$(USER)/Git/euxfel_reduction
SCRATCH = /home/$(USER)/scratch/tmp/2699_098

all : $(SCRATCH)/01_filter.h5 $(SCRATCH)/02_calibrate.h5 $(SCRATCH)/03_sort.h5 $(SCRATCH)/04_fill.h5 $(SCRATCH)/05_integrate.h5 $(SCRATCH)/06_bin.h5 $(SCRATCH)/07_subtract.h5 $(SCRATCH)/08_energy.h5 $(SCRATCH)/09_assemble.h5

$(SCRATCH)/01_filter.h5 : 01_filter.yml detectors.yml $(SCRIPTS)/01_filter.py
	python $(SCRIPTS)/01_filter.py $(SCRATCH) ./
    
$(SCRATCH)/02_calibrate.h5 : $(SCRATCH)/01_filter.h5 02_calibrate.yml 01_filter.yml detectors.yml $(SCRIPTS)/02_calibrate.py
	python $(SCRIPTS)/02_calibrate.py $(SCRATCH) ./
    
$(SCRATCH)/03_sort.h5 : $(SCRATCH)/02_calibrate.h5 03_sort.yml 01_filter.yml detectors.yml $(SCRIPTS)/03_sort.py
	python $(SCRIPTS)/03_sort.py $(SCRATCH) ./
    
$(SCRATCH)/04_fill.h5 : $(SCRATCH)/03_sort.h5 04_fill.yml $(SCRIPTS)/04_fill.py
	python $(SCRIPTS)/04_fill.py $(SCRATCH) ./
    
$(SCRATCH)/05_integrate.h5 : $(SCRATCH)/04_fill.h5 05_integrate.yml 01_filter.yml $(SCRIPTS)/05_integrate.py
	python $(SCRIPTS)/05_integrate.py $(SCRATCH) ./
    
$(SCRATCH)/06_bin.h5 : $(SCRATCH)/05_integrate.h5 06_bin.yml 01_filter.yml $(SCRIPTS)/06_bin.py
	python $(SCRIPTS)/06_bin.py $(SCRATCH) ./  
    
$(SCRATCH)/07_subtract.h5 : $(SCRATCH)/06_bin.h5 07_subtract.yml 01_filter.yml $(SCRIPTS)/07_subtract.py
	python $(SCRIPTS)/07_subtract.py $(SCRATCH) ./
    
$(SCRATCH)/08_energy.h5 : 08_energy.yml $(SCRIPTS)/08_energy.py
	python $(SCRIPTS)/08_energy.py $(SCRATCH) ./
    
$(SCRATCH)/09_assemble.h5 : $(SCRATCH)/06_bin.h5 $(SCRATCH)/08_energy.h5 09_assemble.yml $(SCRIPTS)/09_assemble.py
	python $(SCRIPTS)/09_assemble.py $(SCRATCH) ./
   
# Delete all intermediate files, reset the whole reduction pipeline 
reset : 
	rm -f -v $(SCRATCH)/*.h5
	rm -f -v $(SCRATCH)/*.log

# Delete all but 09_assemble.h5, which holds the final reduced data
clean : 
	rm -f -v $(SCRATCH)/01_filter.h5
	rm -f -v $(SCRATCH)/02_calibrate.h5
	rm -f -v $(SCRATCH)/03_sort.h5
	rm -f -v $(SCRATCH)/04_fill.h5
	rm -f -v $(SCRATCH)/05_integrate.h5
	rm -f -v $(SCRATCH)/06_bin.h5
	rm -f -v $(SCRATCH)/07_subtract.h5    
	rm -f -v $(SCRATCH)/08_energy.h5