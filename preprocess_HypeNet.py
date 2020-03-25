from term_def_seek.extract_term_def_wildly import load_concept_def
import codecs

def convert_raw_hypenet_into_input():
    readfile =codecs.open('/save/wenpeng/datasets/HypeNet/dataset_rnd/test.tsv', 'r', 'utf-8')
    writefile = codecs.open('/save/wenpeng/datasets/HypeNet/dataset_rnd/test.as.input.txt', 'w', 'utf-8')
    pos_size = 0
    neg_size = 0

    for line in readfile:
        parts = line.strip().split('\t')
        term1 = parts[0]
        term2 = parts[1]
        if parts[2] == 'False':
            label = 0
            neg_size+=1
        else:
            label = 1
            pos_size+=1
        def1, _ =load_concept_def(term1)
        def2, _ =load_concept_def(term2)
        writefile.write(str(label)+'\t'+term1+'\t'+def1.replace('\n', ' ')+'\t'+term2+'\t'+def2.replace('\n', ' ')+'\n')
    writefile.close()
    readfile.close()
    print 'pos_size: ', pos_size, 'neg_size: ', neg_size


if __name__ == '__main__':
    convert_raw_hypenet_into_input()
