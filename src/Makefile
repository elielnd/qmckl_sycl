CC = icpx
CFLAGS = -fsycl -pthread
SOURCES = qmckl_basic_functions.cpp qmckl_context.cpp qmckl_memory.cpp test.cpp 

test: $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o test
