# discomt17-pronouns

With the permission of the authors, this code is an adaptation of code from https://github.com/TurkuNLP/smt-pronouns, for their submission to the 2016 shared cross-lingual pronoun prediction shared task at WMT

* Original authors: Juhani Luotolahti, Jenna Kanerva and Filip Ginter
* Date: May 2016
* Date of copy: 28/03/2017

Cf. the article presenting their system description: 
[Juhani Luotolahti, Jenna Kanerva and Filip Ginter. 2016. Cross-Lingual Pronoun Prediction with Deep Recurrent Neural Networks. In Proceedings of the First Conference on Machine Translation. pp. 596–601. Berlin, Germany]( http://www.statmt.org/wmt16/pdf/W16-2353.pdf)

# TODO May 3rd
* HM: Train 2 dep parsing models for each language out of EN, FR, DE and ES, namely one form model and one lemma model. Note these models also require a POS model. 
* HM: Note that the TARGET side has REPLACE_n tokens that are always subject pronouns. One must give them an automatic PRONOUN tag before parsing, as well as a special replace_label label to keep track. Normally they can be treated as subjects.
* HM: Generate a file, aligned to the input. It will contain four columms: SOURCE_form_depidx SOURCE_form_label TARGET_lemma_depidx TARGET_lemma_label
* HM: Note that the TARGET side has REPLACE_n tokens that are always subject pronouns. One must give them an automatic PRONOUN tag before parsing, as well as a special replace_label label to keep track. Normally they can be treated as subjects.

* RB: put data on cluster
* RB: test keras on cluster
* RB: get morphological information from lexica for both source and target sentences. Getting the morph info for target sentences requires mapping the PoS provided to the PoS in the lexicon.
* RB: finish adapting code to take generic features
