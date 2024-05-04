CC = gcc
CFLAGS = -g -Wall -Wextra -pedantic -I.
LDFLAGS =
LIBS = -lm -lSDL2

ifeq ($(OS), Windows_NT)
    CC = x86_64-w64-mingw32-gcc
    LDFLAGS += -L. -L/usr/x86_64-w64-mingw32/lib
    LIBS += -lmingw32
else ifeq ($(shell uname -s), Linux)
	CFLAGS += -I/usr/include
    LDFLAGS += -L. -L/usr/lib/x86_64-linux-gnu
else
	CFLAGS += -I/opt/homebrew/include
    LDFLAGS += -L. -L/opt/homebrew/lib
endif

NeuralNetwork: dense.Layer.o dense.Propagation.o DigitRecogniser.o
	$(CC) $(CPPFLAGS) $(LDFLAGS) -o NeuralNetwork DigitRecogniser.o dense.Layer.o\
		dense.Propagation.o $(LIBS)

dense.Layer.o: dense.Layer.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c dense.Layer.c 

dense.Propagation.o: dense.Propagation.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c dense.Propagation.c 

DigitRecogniser.o: DigitRecogniser.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c DigitRecogniser.c 

clean:
	rm -f *.o NeuralNetwork
