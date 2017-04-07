#!/bin/bash

if [ $# -lt 5 ]; then
    echo "$0: Oh no!! not enough arguments provided."
    echo -e "\nUsage: $0 filename raw_data_folder raw_test_data_folder train_data_folder dev_data_folder test_data_folder\n"
    exit
fi

raw_data_folder=$1
raw_test_data_folder=$2
train_data_folder=$3
dev_data_folder=$4
test_data_folder=$5

# en-fr train

echo "*** train ***"

for lang in "de-en" "en-fr" "en-de" "es-en" ; do
    echo $lang

	[ -d $train_data_folder/$lang ] || mkdir $train_data_folder/$lang

    if [ "$lang" = "es-en" ] ; then
        idfile="es-en"
    fi

    if [ "$lang" = "en-fr" ] || [ "$lang" = "fr-en" ]; then
        idfile="en-fr"
    fi

    if [ "$lang" = "en-de" ] || [ "$lang" = "de-en" ];  then
        idfile="de-en"
    fi

    paste -d"\t" $raw_data_folder/$lang/Europarl.$lang.data $raw_data_folder/$lang/Europarl.$idfile.doc-ids | cut -f 1-6 > $train_data_folder/$lang/Europarl.$lang.data.withids
    paste -d"\t" $raw_data_folder/$lang/IWSLT15.$lang.data $raw_data_folder/$lang/IWSLT15.$idfile.doc-ids > $train_data_folder/$lang/IWSLT15.$lang.data.withids
	if [ "$lang" != "es-en" ]
    then
		paste -d"\t" $raw_data_folder/$lang/NCv9.$lang.data $raw_data_folder/$lang/NCv9.$idfile.doc-ids | cut -f 1-6 > $train_data_folder/$lang/NCv9.$lang.data.withids
	fi

	if [ "$lang" = "es-en" ]
	then
		cat $train_data_folder/$lang/Europarl.$lang.data.withids $train_data_folder/$lang/IWSLT15.$lang.data.withids train_data/NCv9.$lang.data.withids > $train_data_folder/$lang/all.$lang.withids
	else
		cat $train_data_folder/$lang/Europarl.$lang.data.withids $train_data_folder/$lang/IWSLT15.$lang.data.withids > $train_data_folder/$lang/all.$lang.withids
	fi

done 



echo "*** dev ***"


for lang in  "de-en" "en-fr" "en-de" "es-en"; do
    echo $lang

	[ -d $dev_data_folder/$lang ] || mkdir $dev_data_folder/$lang

    if [ "$lang" = "es-en" ] ; then
        idfile="es-en"
    fi
	
    if [ "$lang" = "en-fr" ] || [ "$lang" = "fr-en" ]; then
            idfile="en-fr"
        fi
    if [ "$lang" = "en-de" ] || [ "$lang" = "de-en" ]; then
            idfile="de-en"
    fi

    paste -d"\t" $raw_data_folder/$lang/TEDdev.$lang.data $raw_data_folder/$idfile/TEDdev.$idfile.doc-ids > $dev_data_folder/$lang/TEDdev.$lang.data.withids
    
done


exit 

echo "*** test ***"

for lang in "de-en" "en-fr" "en-de" "es-en"; do
    echo $lang

	[ -d $test_data_folder/$lang ] || mkdir $test_data_folder/$lang

    zcat $raw_test_data_folder/$lang/WMT2016.$lang.data.final.gz | paste -d"\t" - $raw_test_data_folder/$lang/WMT2016.$lang.doc-ids | cut -f 1-6 > $test_data/WMT2016.$lang.data.final.withids

done

