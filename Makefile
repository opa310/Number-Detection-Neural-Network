.PHONY: clean build

CC = gcc
CFLAGS = -g -Wall -Wextra -pedantic -I.
LDFLAGS = -L. -Lbuild/lib/
LIBS = -lm -lSDL2 -lNeuralNetwork

BIN_DIR = build/bin/
LIB_DIR = build/lib/
OBJ_DIR = build/obj/

# File with Main defined
OBJ_EXCLUDE = DigitRecogniser.o

# Source files
SRC_FILES = $(wildcard *.c)
OBJ_FILES = $(patsubst %.c, $(OBJ_DIR)%.o, $(SRC_FILES))
OBJ_INCLUDE = $(filter-out $(OBJ_DIR)$(OBJ_EXCLUDE), $(OBJ_FILES))

ifeq ($(OS), Windows_NT)
    CC = x86_64-w64-mingw32-gcc
    LDFLAGS += -L/usr/x86_64-w64-mingw32/lib
    LIBS += -lmingw32
else ifeq ($(shell uname -s), Linux)
    CFLAGS += -I/usr/include
    LDFLAGS += -L/usr/lib/x86_64-linux-gnu
else ifeq ($(shell uname -s), Darwin)
    CFLAGS += -I/opt/homebrew/include
    LDFLAGS += -L/opt/homebrew/lib
endif

# Rule to build the final binary
$(BIN_DIR)NeuralNetwork: $(LIB_DIR)libNeuralNetwork.a $(OBJ_DIR)DigitRecogniser.o | build
	$(CC) $(LDFLAGS) -o $@ $(OBJ_DIR)DigitRecogniser.o $(LIBS)

# Rule to build the static library excluding the specified object file
$(LIB_DIR)libNeuralNetwork.a: $(OBJ_INCLUDE) | build
	ar rcs $@ $^

# Rule to compile source files into object files in the object directory
$(OBJ_DIR)%.o: %.c | build
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	rm -rf build Digit-Recogniser.csv


build:
	mkdir -p $(BIN_DIR) $(LIB_DIR) $(OBJ_DIR)
