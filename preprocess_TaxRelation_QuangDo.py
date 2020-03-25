
import codecs
from term_def_seek.extract_term_def_wildly import load_concept_def

root = '/save/wenpeng/datasets/TaxonomicRelationData/DataII/'

def preprocess_files():
    files = ['train', 'test']
    wfiles = ['train.in.def.txt', 'test.in.def.txt']

    for i in range(len(files)):
        readfile = codecs.open(root+files[i], 'r','utf-8')
        writefile  = codecs.open(root+wfiles[i], 'w','utf-8')
        line_co=0
        for line in readfile:
            parts=line.strip().split('\t')
            if len(parts) == 4:
                line_co+=1
                label = parts[0].strip()
                term1 = parts[2]
                term2 = parts[3]
                term1_def, source1 = load_concept_def(term1)
                term2_def, source2 = load_concept_def(term2)
                writefile.write(label+'\t'+term1+'\t'+' '.join(term1_def.split())+'\t'+term2+'\t'+' '.join(term2_def.split())+'\n')
        writefile.close()
        readfile.close()
        print 'parse over: ', root+files[i]

if __name__ == '__main__':
    preprocess_files()
