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

	filtered=".filtered"
    if [ "$lang" = "es-en" ] ; then
        idfile="es-en"
		filtered=""
    fi
	
    if [ "$lang" = "en-fr" ] ; then
        idfile="en-fr"
    fi

    if [ "$lang" = "en-de" ] || [ "$lang" = "de-en" ];  then
        idfile="de-en"
    fi

    zcat $raw_data_folder/$lang/Europarl.$lang.data$filtered.gz | paste -d"\t" - $raw_data_folder/$idfile/Europarl.$idfile.doc-ids |  cut -f 1-6 | gzip > $train_data_folder/$lang/Europarl.$lang.data$filtered.withids.gz
    zcat $raw_data_folder/$lang/IWSLT15.$lang.data$filtered | paste -d"\t" - $raw_data_folder/$idfile/IWSLT15.$idfile.doc-ids | gzip > $train_data_folder/$lang/IWSLT15.$lang.data$filtered.withids.gz
	if [ "$lang" != "es-en" ]
    then
		zcat $raw_data_folder/$lang/NCv9.$lang.data$filtered.gz | paste -d"\t" - $raw_data_folder/$idfile/NCv9.$idfile.doc-ids | cut -f 1-6 | gzip > $train_data_folder/$lang/NCv9.$lang.data$filtered.withids.gz
	fi

	if [ "$lang" != "es-en" ]
	then
		zcat $train_data_folder/$lang/Europarl.$lang.data$filtered.withids.gz $train_data_folder/$lang/IWSLT15.$lang.data$filtered.withids.gz train_data/$lang/NCv9.$lang.data$filtered.withids.gz | gzip > $train_data_folder/$lang/all.$lang.withids.gz
	else
		zcat $train_data_folder/$lang/Europarl.$lang.data$filtered.withids.gz $train_data_folder/$lang/IWSLT15.$lang.data$filtered.withids.gz | gzip > $train_data_folder/$lang/all.$lang.withids.gz
	fi

done 



echo "*** dev ***"


for lang in  "de-en" "en-fr" "en-de" "es-en"; do
    echo $lang

	[ -d $dev_data_folder/$lang ] || mkdir $dev_data_folder/$lang

    if [ "$lang" = "es-en" ] ; then
        idfile="es-en"
		filtered=""
	else
		filtered=".filtered"
    fi
	
    if [ "$lang" = "en-fr" ] || [ "$lang" = "fr-en" ]; then
            idfile="en-fr"
        fi
    if [ "$lang" = "en-de" ] || [ "$lang" = "de-en" ]; then
            idfile="de-en"
    fi

    zcat $raw_data_folder/$lang/TEDdev.$lang.data$filtered | paste -d"\t" - $raw_data_folder/$idfile/TEDdev.$idfile.doc-ids | gzip > $dev_data_folder/$lang/TEDdev.$lang.data$filtered.withids.gz
    
done


echo "*** test ***"

for lang in "de-en" "en-fr" "en-de" "es-en"; do
    echo $lang

	if [ "$lang" = "es-en" ] ; then
		filtered=""
	else
		filtered=".filtered"
    fi

	[ -d $test_data_folder/$lang ] || mkdir $test_data_folder/$lang

    zcat $raw_test_data_folder/$lang/WMT2016.$lang.data$filtered.final.gz | paste -d"\t" - $raw_test_data_folder/$lang/WMT2016.$lang.doc-ids | cut -f 1-6 | gzip > $test_data_folder/WMT2016.$lang.data$filtered.final.withids.gz

done

