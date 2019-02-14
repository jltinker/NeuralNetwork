hd = $(HOME)/cosmo/lib
LIB = -lm -L${hd} -lcutil #-fopenmp


CC = gcc
CFLAGS = -O2 #-fopenmp

OBJS1 = neural_network.o
ANN:	$(OBJS1)
	$(CC) -o $@ $(OBJS1) $(LIB)
	cp -f $@ $(HOME)/exec/$@

OBJS2 = neural_network_2layer.o
ANN2L:	$(OBJS2)
	$(CC) -o $@ $(OBJS2) $(LIB)
	cp -f $@ $(HOME)/exec/$@


clean:
	rm -f *.o
