
import argparse, os

def read_results_file(filename):
    start_calc = False
    label2score = {}
    with open(filename, "r", encoding="utf-8") as fp:
        for line in fp:
            if start_calc:
                if line.strip()=="": break # end of labels
                lp = line.strip().split()
                label, p, r, f1 = lp[0], lp[7], lp[13], lp[16]
                label2score[label] = (p, r, f1)
                
            # identify place in file to start looking
            if "Results for the individual labels" in line:
                start_calc = True

    return label2score

def read_all_results(foldername, langpair):

    sub2scores = {}
    for fol in os.listdir(foldername):
        if os.path.isdir(foldername+"/"+fol):
            for fol2 in os.listdir(foldername+"/"+fol):
                if fol2==langpair:
                    for fic in os.listdir(foldername+"/"+fol+"/"+fol2):
                        if "primary.score" in fic:
                            filename=foldername+"/"+fol+"/"+fol2+"/"+fic
                            sub2scores[fol] = read_results_file(filename)
    return sub2scores

def print_line(label2scores, scoretype):
    for label in sorted(label2scores):
        os.sys.stdout.write(" | "+label2scores[label][scoretype])
    os.sys.stdout.write(" |\n")

def print_tables(sub2scores):
    labels = sorted(list(sub2scores[list(sub2scores.keys())[0]].keys()))
    
    os.sys.stdout.write("\n| Precision | "+" | ".join(labels)+" |\n")
    os.sys.stdout.write("| --- "*(len(labels)+1)+" | \n")
    for sub in sub2scores:
        os.sys.stdout.write("| "+sub)
        print_line(sub2scores[sub], 0) # precision
        
    os.sys.stdout.write("\n| Recall | "+" | ".join(labels)+" |\n")
    os.sys.stdout.write("| --- "*(len(labels)+1)+" | \n")
    for sub in sub2scores:
        os.sys.stdout.write("| "+sub)
        print_line(sub2scores[sub], 1) # recall
        
    os.sys.stdout.write("\n| F1 score | "+" | ".join(labels)+" |\n")
    os.sys.stdout.write("| --- "*(len(labels)+1)+" | \n")
    for sub in sub2scores:
        os.sys.stdout.write("| "+sub)
        print_line(sub2scores[sub], 2) # fscore
    os.sys.stdout.write("\n")


if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("results_folder")
    args = argparser.parse_args()

    for langpair in ["de-en", "en-de", "en-fr", "fr-en"]:
        os.sys.stdout.write("\n"+langpair+"\n")
        sub2scores = read_all_results(args.results_folder, langpair)
        print_tables(sub2scores)

