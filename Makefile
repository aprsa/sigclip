all:	clean sigclip

sigclip:
	gcc sigclip.c -o sigclip -lgsl -lgslcblas -lm -Wall -g

clean:
	rm -rf sigclip

