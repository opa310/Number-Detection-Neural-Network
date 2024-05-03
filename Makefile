CC = gcc 
CFLAGS = -g -Wall -Wextra -pedantic -I.
LDFLAGS = 
LIBS = -lm -lSDL2

NeuralNetwork: dense.Layer.o dense.Propagation.o DigitRecogniser.o
	$(CC) $(CPPFLAGS) -L. -L/opt/homebrew/lib -o NeuralNetwork DigitRecogniser.o dense.Layer.o\
		dense.Propagation.o $(LIBS)

dense.Layer.o: dense.Layer.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c dense.Layer.c 

dense.Propagation.o: dense.Propagation.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c dense.Propagation.c 

DigitRecogniser.o: DigitRecogniser.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -I/opt/homebrew/include -c DigitRecogniser.c 


clean:
	rm -f *.o NeuralNetwork
