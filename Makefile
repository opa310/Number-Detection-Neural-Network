CC = gcc
CFLAGS = -g
CPPFLAGS = -Wall -Wextra -pedantic
LIBS = -lm

NeuralNetwork: dense.Layer.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -L. -o NeuralNetwork dense.Layer.o $(LIBS)

dense.Layer.o: dense.Layer.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c dense.Layer.c 

clean:
	rm -f *.o NeuralNetwork
