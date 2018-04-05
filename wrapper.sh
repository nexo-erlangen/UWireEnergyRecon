#!/bin/bash

timestamp() {
	date +"%y%m%d-%H%M"
}
echo $(timestamp)

#check ob auf GPU mit Hilfe von $HOST ?
PWD='/home/vault/capm/sn0515/PhD/Th_U-Wire'
DataOLD=$PWD/Data_MC

array=( $* )
for((i=0; i<$#; i++)) ; do
	case "${array[$i]}" in
		--test) TEST="true";;
		--prepare) PREPARE="true";;
		--resume) RESUME="true";;
		-model) PARENT=${array[$i+1]};;
	esac
done 

if [[ $TEST == "true" ]] ||  [[ $PREPARE == "true" ]] ; then
	FolderOUT=$PWD/MonteCarlo/Dummy
else
	if [[ $RESUME=="true" ]] && [[ ! -z $PARENT ]] ; then
		if [ -d $PARENT ] ; then
			FolderOUT=${PARENT}$(timestamp)
		else
			echo "Model Folder (${PARENT}) does not exist" ; exit 1
		fi
	else
		FolderOUT=$PWD/MonteCarlo/$(timestamp)
	fi
fi

mkdir -p $FolderOUT && cp $PWD/script_*.py $FolderOUT && cp $PWD/Scripts/script_plot.py $FolderOUT

DataNEW=$DataOLD
#if [[ ! -z $TMPDIR ]] && ( [[ $TEST != "true" ]] || [[ $PREPARE == "true" ]] ) ; then
#	DataNEW=$TMPDIR/data
#	mkdir -p $DataNEW && cp -ru $DataOLD/*[$COPYONLY]* $DataNEW
#else
#	DataNEW=$DataOLD
#	echo "\$TMPDIR is not defined/used. Getting Files from ${DataOLD}"
#fi

echo
if [[ $PREPARE == "true" ]] ; then
	echo "(python $FolderOUT/script_train.py -in $DataNEW -out $FolderOUT ${@:1}) | tee $FolderOUT/log.dat"
else
	(python $FolderOUT/script_train.py -in $DataNEW -out $FolderOUT ${@:1}) | tee $FolderOUT/log.dat
	echo $(timestamp)
fi

if [[ ! -z $TMPDIR ]] && [[ $TEST != "true" ]] && [[ $PREPARE != "true" ]] ; then
	echo "(rm -r $TMPDIR/data)"
fi
