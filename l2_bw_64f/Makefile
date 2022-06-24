GENCODE_SM75 := -gencode=arch=compute_75,code=\"sm_75,compute_75\"

CUOPTS = $(GENCODE_SM75)


CC := nvcc

INCLUDE := 
LIB := 

SRC = l2_bw_64f.cu

EXE = l2_bw_64f

NVCC_FLGAS = -Xptxas -dlcm=cg -Xptxas -dscm=wt

release:
	$(CC) $(NVCC_FLGAS) $(CUOPTS) $(SRC) -o $(EXE) -I$(INCLUDE) -L$(LIB) -lcudart

clean:
	rm -f *.o; rm -f $(EXE)

run:
	./$(EXE)

profile:
	nvprof ./$(EXE)

events:
	nvprof  --events elapsed_cycles_sm ./$(EXE)
