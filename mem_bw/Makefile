GENCODE_SM61 := -gencode=arch=compute_61,code=\"sm_61,compute_61\"
GENCODE_SM70 := -gencode=arch=compute_70,code=\"sm_70,compute_70\"
GENCODE_SM75 := -gencode=arch=compute_75,code=\"sm_75,compute_75\"

CUOPTS = $(GENCODE_SM75)


CC := nvcc

INCLUDE := 
LIB := 

SRC = mem_bw.cu

EXE = mem_bw

NVCC_FLGAS = -Xptxas -dlcm=cg -Xptxas -dscm=wt

release:
	$(CC) $(NVCC_FLGAS)  $(CUOPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) -lcudart

run:
	./$(EXE)

profile:
	nvprof ./$(EXE)

profileall:
	nvprof --concurrent-kernels off --print-gpu-trace -u us --metrics all --demangling off --csv --log-file data.csv ./$(EXE)

events:
	nvprof  --events elapsed_cycles_sm ./$(EXE)
