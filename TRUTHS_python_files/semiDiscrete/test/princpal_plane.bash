#!/bin/bash 


COMMON="-rsl1 0.2 -rsl2 0.1 -rsl3 0.03726 -rsl4 0.002426 -cab 75.0 -cw 0.01 -cp 0.001 -cc 0.001 -N 1"

VZA=35
SZA=30
VAA=0
SAA=0

NADIM_OPTS="$COMMON -lad 5 -hc 3"
KUUSK_OPTS="$COMMON -thm 45"


W=800

for L in 0.1 1 2 3 4 5 7 9 ; do
#for L in 0.1 ; do

	> pplane.L_$L.dat
	> pplane_hs_test.L_$L.dat

	#for W in $(gawk 'BEGIN{for(i=400;i<=2500;i++)print i}') ; do
	for VZA in $(gawk 'BEGIN{for(i=-85;i<=85;i+=1)print i}') ; do

		N=$(echo -e "1 1 $W\n$VZA $VAA $SZA $SAA"|../nadim -LAI $L $NADIM_OPTS|gawk 'NR==2{print $NF}')
		Nx=$(echo -e "1 1 $W\n$VZA $VAA $SZA $SAA"|../nadim_nohotspot -LAI $L $NADIM_OPTS|gawk 'NR==2{print $NF}')
		#K=$(echo -e "1 1 $W\n$VZA $VAA $SZA $SAA"|kuusk -L $L $KUUSK_OPTS|gawk 'NR==2{print $NF}')

		#echo $VZA $N $K >> pplane.L_$L.dat
		echo $VZA $N $Nx >> pplane_hs_test.L_$L.dat
		
	done

done
