.PHONY: clean build 

CC = gcc
CFLAGS = -g -Wall -Wextra -pedantic -I.
LDFLAGS =
LIBS = -lm -lSDL2

BIN_DIR = build/bin/
LIB_DIR = build/lib/
OBJ_DIR = build/obj/


ifeq ($(OS), Windows_NT)
    CC = x86_64-w64-mingw32-gcc
    LDFLAGS += -L. -L/usr/x86_64-w64-mingw32/lib
    LIBS += -lmingw32
else ifeq ($(shell uname -s), Linux)
CFLAGS += -I/usr/include
    LDFLAGS += -L. -L/usr/lib/x86_64-linux-gnu
else ifeq ($(shell uname -s), Darwin)
CFLAGS += -I/opt/homebrew/include
    LDFLAGS += -L. -L/opt/homebrew/lib
endif


$(BIN_DIR)NeuralNetwork: $(OBJ_DIR)dense.Layer.o $(OBJ_DIR)dense.Propagation.o $(OBJ_DIR)DigitRecogniser.o
	$(CC) $(LDFLAGS) -o $(BIN_DIR)NeuralNetwork $(OBJ_DIR)DigitRecogniser.o $(OBJ_DIR)dense.Layer.o\
		$(OBJ_DIR)dense.Propagation.o $(LIBS)

$(OBJ_DIR)dense.Layer.o: dense.Layer.c build
	$(CC) $(CFLAGS) -c dense.Layer.c -o $(OBJ_DIR)dense.Layer.o

$(OBJ_DIR)dense.Propagation.o: dense.Propagation.c build
	$(CC) $(CFLAGS) -c dense.Propagation.c -o $(OBJ_DIR)dense.Propagation.o  

$(OBJ_DIR)DigitRecogniser.o: DigitRecogniser.c build
	$(CC) $(CFLAGS) -c DigitRecogniser.c -o $(OBJ_DIR)DigitRecogniser.o 

clean:
	rm -rf build Digit-Recogniser.csv

build:
	mkdir -p build/{bin,lib,obj}
