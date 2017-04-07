#!/bin/bash


#------------------------------------------------------------
raw_data="/vol/work2/bawden/DiscoMT2017/raw_data"
raw_test_data="/vol/work2/bawden/DiscoMT2017/raw_test_data"
train_data="/vol/work2/bawden/DiscoMT2017/train_data"
dev_data="/vol/work2/bawden/DiscoMT2017/dev_data"
test_data="/vol/work2/bawden/DiscoMT2017/test_data"
#------------------------------------------------------------

# en-fr train

echo "*** train ***"

for lang in "de-en" "en-fr" "en-de" "sp-en"
    do
    echo $lang

    if [ "$lang" = "en-fr" ] || [ "$lang" = "fr-en" ]
    then
        idfile="en-fr"
    fi

    if [ "$lang" = "en-de" ] || [ "$lang" = "de-en" ]
    then
        idfile="de-en"
    fi

    if [ "$lang" = "fr-en" ]
    then 
        filtered=""
    else
        filtered=".filtered"
    fi


    paste -d"\t" $raw_data/$lang/Europarl.$lang.data$filtered $raw_data/$lang/Europarl.$idfile.doc-ids | cut -f 1-6 > $train_data/$lang/Europarl.$lang.data$filtered.withids
    paste -d"\t" $raw_data/$lang/IWSLT15.$lang.data$filtered $raw_data/$lang/IWSLT15.$idfile.doc-ids > $train_data/$lang/IWSLT15.$lang.data$filtered.withids
    paste -d"\t" $raw_data/$lang/NCv9.$lang.data$filtered $raw_data/$lang/NCv9.$idfile.doc-ids | cut -f 1-6 > $train_data/$lang/NCv9.$lang.data$filtered.withids
    
    cat $train_data/$lang/Europarl.$lang.data$filtered.withids $train_data/$lang/IWSLT15.$lang.data$filtered.withids $train_data/$lang/NCv9.$lang.data$filtered.withids > $train_data/$lang/all.$lang$filtered.withids

done 



echo "*** dev ***"

for lang in  "de-en" "en-fr" "en-de" 
    do
    echo $lang

    if [ "$lang" = "en-fr" ] || [ "$lang" = "fr-en" ]
        then
            idfile="en-fr"
        fi
    if [ "$lang" = "en-de" ] || [ "$lang" = "de-en" ]
        then
            idfile="de-en"
        fi
    
    if [ "$lang" = "fr-en" ]
    then 
        filtered=""
    else
        filtered=".filtered"
    fi

    paste -d"\t" $raw_data/$lang/TEDdev.$lang.data$filtered $raw_data/$lang/TEDdev.$idfile.doc-ids > $dev_data/$lang/TEDdev.$lang.data$filtered.withids
    

done


echo "*** test ***"

for lang in "de-en" "en-fr" "en-de" 
    do
    echo $lang

    if [ "$lang" = "fr-en" ]
    then 
        filtered=""
    else
        filtered=".filtered"
    fi


    zcat $raw_test_data/$lang/WMT2016.$lang.data$filtered.final.gz | paste -d"\t" - $raw_test_data/$lang/WMT2016.$lang.doc-ids | cut -f 1-6 > $test_data/$lang/WMT2016.$lang.data$filtered.final.withids
    

done

