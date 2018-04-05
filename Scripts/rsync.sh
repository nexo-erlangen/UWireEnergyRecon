cd #!/bin/bash
#Ordner auf dem HPC_DATA Server am ECAP
#SRC_DIR=rsync://131.188.161.205/ECAP_HPC_NEXO/DeepLearning/Th_Study_Uwire/Co60_Wfs_SS+MS_S5_Data/
#DEST_DIR=/home/woody/capm/sn0515/PhD/Th_U-Wire/Co60_Wfs_SS+MS_S5_Data/
SCRATCH=/scratchssd/sn0515/Co60_WFs_S5_MC/

DEST_DIR=rsync://131.188.161.205/ECAP_HPC_NEXO/DeepLearning/Th_Study_Uwire/Co60_Wfs_SS+MS_S5_Data/
SRC_DIR=/home/woody/capm/sn0515/PhD/Th_U-Wire/Co60_Wfs_SS+MS_S5_Data/

echo
echo "Copying file:"
#Kopieren und ausfuehren
for i in $(rsync --list-only $SRC_DIR/*.hdf5 | awk '{print $5}') ; do
	if [ $i = "." ] || [ $i == ".*" ]
	then
		continue
	fi
	#rsync -au $SRC_DIR$i $SCRATCH
	#echo -e "${SRC_DIR}${i} \t --->  ${SCRATCH}${i}"
	echo -e "${SRC_DIR}${i} \t --->  ${DEST_DIR}"
	rsync -Pau $SRC_DIR$i $DEST_DIR
	
	
done

echo
#(python /home/vault/capm/sn0515/PhD/Th_U-Wire/Scripts/preprocess_MC.py -in ${SCRATCH} -out ${DEST_DIR})

echo
echo "Removing temporary Files:"
for i in $SCRATCH* ; do
	if [ $i = "." ] || [ $i == ".*" ]
	then
		continue
	fi
	[[ -e $i ]] || continue

	echo -e "removing \t ${i}"

	# Clean up
	#rm $SCRATCH$i
done
