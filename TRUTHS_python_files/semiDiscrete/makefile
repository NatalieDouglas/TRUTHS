CC = gcc
#for linux use:
#FF = f77
#for solaris use:
FF=gfortran

CFLAGS = -Wall -g
LIBS = -lm 
#HOME = /home/mdisney
#INCLS = -I/usr/local/include
#INCLS += -I/usr/include
INCLS = -I./

OUT = nadim

FOBJECTS = nadimbrf.o nadimtools.o prospect.redux.o nadimbrfe.o
FOBJECTS_NOHS = nadimbrf.o nadimtools_nohotspot.o prospect.redux.o nadimbrfe.o
COBJECTS = nadim.o prospect_interface.o price_soil.o


$(OUT):		${FOBJECTS} $(COBJECTS)
		${FF} ${CFLAGS} ${FOBJECTS} ${COBJECTS} -o ${OUT} 

nohotspot:	${FOBJECTS_NOHS} $(COBJECTS)
		${FF} ${CFLAGS} ${FOBJECTS_NOHS} ${COBJECTS} -o ${OUT}_nohotspot 



.c.o: $<
		$(CC) ${INCLS} $(CFLAGS) -c $<

.f.o: $<
		$(FF) $(CFLAGS) -c $<

clean:
		\rm -f *.o *~ *%

#install:
#		cp prospect $(HOME)/bin/$(ARCH)/prospect.redux

