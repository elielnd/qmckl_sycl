CC = icpx
CPPFLAGS = -fsycl
LDFLAGS = -pthread 
OUTPUT = qmckl_gpu
SRC_DIR = src
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(SRC:$(SRC_DIR)/%.cpp=$(SRC_DIR)/%.o)


$(OUTPUT) : $(OBJ)
	$(CC) $(CPPFLAGS) -o $@ $^ $(LDFLAGS)


$(SRC_DIR)/%.o : $(SRC_DIR)/%.cpp
	$(CC) -c $(CPPFLAGS) $< -o $@ $(LDFLAGS)


clean :
	rm -f $(OBJ) $(OUTPUT)