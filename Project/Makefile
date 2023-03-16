INCLUDE_DIRS = -I/usr/include/opencv4

LIB_DIRS = 
CC=g++

CDEFS=
CFLAGS= -O3 -g $(INCLUDE_DIRS) $(CDEFS) 
LIBS= -L/usr/lib -lopencv_core -lopencv_flann -lopencv_video -lrt -pthread -lpthread

HFILES= 
CFILES= main.cpp
SRCS= ${HFILES} ${CFILES}
OBJS= ${CFILES:.cpp=.o}

all:	main

clean:
	-rm -f *.o *.d
	-rm -f main

capture2: capture2.o
	$(CC) $(LDFLAGS) $(CFLAGS) -o $@ $@.o `pkg-config --libs opencv4` $(LIBS)

depend:

.cpp.o: $(SRCS)
	$(CC) $(CFLAGS) -c $<
