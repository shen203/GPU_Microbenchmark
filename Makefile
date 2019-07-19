
BASE_DIR := $(shell pwd)
BIN_DIR := $(BASE_DIR)/bin

all: 	
	mkdir -p $(BIN_DIR)
	cd l1_bw_32f;		make;	cp l1_bw_32f $(BIN_DIR)
	cd l1_bw_64f; 		make;	cp l1_bw_64f $(BIN_DIR)
	cd l1_lat_32f 		make;	cp l1_lat_32f $(BIN_DIR)
	cd l1_lat_64f;		make;	cp l1_lat_64f $(BIN_DIR)
	cd l2_bw_32f;  		make;	cp l2_bw_32f $(BIN_DIR)
	cd l2_bw_64f; 		make;	cp l2_bw_64f $(BIN_DIR)
	cd l2_lat_32f; 		make;	cp l2_lat_32f $(BIN_DIR)
	cd l2_lat_64f; 		make;	cp l2_lat_64f $(BIN_DIR)
	cd mem_bw; 		make;	cp mem_bw $(BIN_DIR)
	cd mem_lat; 		make;	cp mem_lat $(BIN_DIR)

clean:
	cd $(BIN_DIR); rm -f *
	for dir in $(BASE_DIR) ; do cd $$dir ; make clean ; cd .. ; done
