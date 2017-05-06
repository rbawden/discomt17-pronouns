import argparse, gzip, re, os

def read_lexicon(fname):
    if ".gz" in fname: fp = gzip.open(fname, "rt", encoding="utf8")
    else: fp = open(fname, "rt", encoding="utf8")

    lexicon = {}
    for line in fp:
        cols = line.strip("\n").split("\t")
        if cols[3]=="": continue
        
        if cols[0] not in lexicon: lexicon[cols[0]] = {}
        if cols[1] not in lexicon[cols[0]]: lexicon[cols[0]][cols[1]] = []

        lexicon[cols[0]][cols[1]].append(cols[3])
    fp.close()
    return lexicon


def map_pos_de(pos):
    return {".": "(parentf|parento|poncts|ponctw)",
            "ADJ": "ADJ",
            "ADP": "(APPO|APPR|APZR)",
            "ADV": "ADV",
            "CONJ": "(KON|KOKOM|KOUI|KOUS)",
            "DET": "ART",
            "NOUN": "N",
            "NUM": "CARD",
            "PRON": "PRO.*",
            "PRT": "(PTKA|PTKANT)",
            "VERB": "(V|MOD|AUX|PP)",
            "X": "(FN|ITJ|PTKA|PTKANT|PTKNEG|PTKVZ|PTKZU)"}.get(pos, pos) # return pos as default

def map_pos_en(pos):
    return {".": "(parentf|parento|poncts|ponctw|epsilon)",
            "ADJ": "(A|adjPref|adjSuff|R)",
            "ADP": "(A|R|S)",
            "ADV": "(adv|advPref|R|S)",
            "CONJ": "C",
            "DET": "(det|D)",
            "NOUN": "(nc?|np|N|P)",
            "NUM": "CARD",
            "PRON": "pro",
            "PRT": "(PTKA|PTKANT|S)",
            "VERB": "(V|MOD|AUX|PP)",
            "X": "I"}.get(pos, pos) # return pos as default

def map_pos_fr(pos):
    return {".": "(parentf|parento|poncts|ponctw)",
            "ABR": "(np|nc)",
            "ADJ": "(adj|adjPref)",
            "ADV": "(adv|pri|pres|csu|advneg|advPred)",
            "KON": "(coo|que_restr|que|csu)",
            "DET": "det",
            "INT": "(pres|nc)", 
            "NAM": "np",
            "NOM": "nc",
            "NUM": "CARD",
            "PRON": "(cl(a|ar|d|dr|g|l|n|neg|r)|que|pro|pri|prel|ilimp|ce|caimp)",
            "PRP": "prep",
            "PUN": "(poncts|ponctw|epsilon)",
            "VER": "(v|auxAvoir|auxEtre)",
            "SYM":  "(poncts|ponctw|epsilon)"
            }.get(pos, pos) # return pos as default

def map_pos(lang):
    if lang=="de": return map_pos_de
    elif lang=="fr": exit("raah")
    elif lang=="es": exit("raah")
    elif lang=="en": exit("raah")
    

def parse_morph(morph):
    dictmorph = {}
    for el in morph.split("."):
        if el in ["subj", "ind", "inf"]: dictmorph["mood"] = el
        elif el in ["pres", "ppart", "past"]: dictmorph["tense"] = el
        elif el in ["sg", "pl"]: dictmorph["number"] = el
        elif el in "123": dictmorph["person"] = el
        elif el in ["fem", "masc", "neu"]: dictmorph["gender"] = el
        elif el in ["acc", "dat", "gen", "nom"]: dictmorph["case"] = el
        elif el in ["long", "short", "short_sg", "short_pl"]: dictmorph["longshort"] = el
        elif el in ["plain", "noagr", "primary", "secondary"]: continue
        elif el in ["super"]: dictmorph["compar"] = el
        elif el in ["imp"]: dictmorph["imp"] = el
        else:
            continue
    return dictmorph
        
def get_morph_from_lexicon(entry, constrainpos = "[.*]",
                           keep = ["tense", "gender", "number", "case", "person", "mood", "pos"]):
    posmorph = {"pos":set([])}
    for pos in entry:
        posmorph["pos"].add(pos)
        if not re.match("^("+"|".join(constrainpos)+")$", pos): continue
        for morph in entry[pos]:
            dictmorph = parse_morph(morph)
            for info in keep:
                if info in dictmorph:
                    if info not in posmorph: posmorph[info] = set([])
                    posmorph[info].add(dictmorph[info])
                    

    # print("**** posmorph", posmorph)
    return ".".join(["-".join(sorted(list(posmorph[x]))) for x in sorted(posmorph)])

def get_fine_pos(fname, colnum, lexicon, lang, wordpos=False):
    posmap = map_pos(lang)

    # decide what morphinfo to keep 
    keep = ["gender", "person", "number", "pos"]
    
    if ".gz" in fname: fp = gzip.open(fname, "rt", encoding="utf-8")
    else: fp = open(fname, "r", encoding="utf-8")

    for line in fp:
        l=0
        for word in line.strip("\n").split("\t")[colnum-1].split(" "):

            # get pos and word
            pos, finepos = [".*?"], "NA"
            if wordpos:
                if "|" in word:
                    word, pos = word.split("|")
                    pos = posmap(pos)
                    
            # if word not recognised, just send old POS (if present)
            if word not in lexicon:
                continue

            # otherwise, look up in the lexicon
            finepos = get_morph_from_lexicon(lexicon[word], pos, keep)
            if finepos=="": finepos="."

            if l>0:os.sys.stdout.write(" ")
            l+=1
            os.sys.stdout.write(finepos)
        os.sys.stdout.write("\n")
           
                

if __name__=="__main__":
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_fname", help="data file")
    argparser.add_argument("lang", choices=["en", "es", "de", "fr"])
    argparser.add_argument("column_number", type=int, help="column number of which to get POS (from 1)")
    argparser.add_argument("mlex", help="mlex lexicon") # form, token, lemma
    argparser.add_argument("-p", "--pos", help="POS also available", action="store_true", default=False)
    args = argparser.parse_args()

    lexicon = read_lexicon(args.mlex)
    get_fine_pos(args.data_fname, args.column_number, lexicon, args.lang, args.pos)

    


