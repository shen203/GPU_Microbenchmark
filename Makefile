
BASE_DIR := $(shell pwd)
BIN_DIR := $(BASE_DIR)/bin

all: 	
	mkdir -p $(BIN_DIR)
	cd l1_bw_32f;		make;	cp l1_bw_32f $(BIN_DIR)
	cd l1_bw_64f; 		make;	cp l1_bw_64f $(BIN_DIR)
	cd l1_bw_128;		make;	cp l1_bw_128 $(BIN_DIR)
	cd l1_lat; 		make;	cp l1_lat $(BIN_DIR)
	cd l2_bw_32f;  		make;	cp l2_bw_32f $(BIN_DIR)
	cd l2_bw_64f; 		make;	cp l2_bw_64f $(BIN_DIR)
	cd l2_bw_128; 		make;	cp l2_bw_128 $(BIN_DIR)
	cd l2_lat; 		make;	cp l2_lat $(BIN_DIR)
	cd mem_bw; 		make;	cp mem_bw $(BIN_DIR)
	cd mem_lat; 		make;	cp mem_lat $(BIN_DIR)
	cd MaxFlops; 		make;	cp MaxFlops $(BIN_DIR)
	cd shared_lat; 		make;	cp shared_lat $(BIN_DIR)
	cd shared_bw; 		make;	cp shared_bw $(BIN_DIR)

clean:
	cd $(BIN_DIR); rm -f *
