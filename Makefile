CC = gcc 
CFLAGS = -g
CPPFLAGS = -Wall -Wextra -pedantic
LIBS = -lm

NeuralNetwork: dense.Layer.o dense.Propagation.o
	$(CC) $(CPPFLAGS) $(CFLAGS) -L. -o NeuralNetwork dense.Layer.o dense.Propagation.o $(LIBS)

dense.Layer.o: dense.Layer.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c dense.Layer.c 

dense.Propagation.o: dense.Propagation.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I. -c dense.Propagation.c 

clean:
	rm -f *.o NeuralNetwork
