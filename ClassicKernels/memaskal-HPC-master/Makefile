CC=nvcc
CXXFLAGS=-O0
EXTRAFLAGS=-arch=sm_20

# This variable is used as the action name.
FILE1=erwt1
FILE2=erwt2
FILE3=erwt3

$(FILE1): ./src/$(FILE1).cu
	$(CC) -lcublas -o $(FILE1).out $(CXXFLAGS) $(EXTRAFLAGS) ./src/$(FILE1).cu
$(FILE2): ./src/$(FILE2).cu
	$(CC) -o $(FILE2).out $(CXXFLAGS) $(EXTRAFLAGS) ./src/$(FILE2).cu
$(FILE3): ./src/$(FILE3).cu
	$(CC) -o $(FILE3).out $(CXXFLAGS) $(EXTRAFLAGS) ./src/$(FILE3).cu
clean:
	#delete executables
	$(RM) $(FILE1).out
	$(RM) $(FILE2).out
	$(RM) $(FILE3).out