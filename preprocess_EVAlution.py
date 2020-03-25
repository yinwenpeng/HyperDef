from term_def_seek.common_functions import extract_word_deflist_given_POS,extract_word_top_def_from_wordnet,replace_punctuation_in_text_by_whitespace,extract_word_all_def_list



def Extract_Top_Def():
    #given POS tag
    folder = '/save/wenpeng/datasets/UnsupervisedHypernymy/'
    files = ['BLESS.test','EVALution.test','LenciBenotto.test','Weeds.test']
    '''
    shadow-n        follow-n        True    hyper
    coffee-n        good-n  False   attrib
    space-v place-v False   synonym
    ceiling-n       floor-n False   antonym
    '''
    outfiles = ['BLESS.test.top.definition','EVALution.test.top.definition','LenciBenotto.test.top.definition','Weeds.test.top.definition']

    for i in range(len(files)):
        count=0
        hyper_count = 0
        remove_size =0
        readfile = open(folder+files[i], 'r')
        writefile = open(folder+outfiles[i], 'w')
        for line in readfile:
            parts = line.strip().split()
            word1_pos = parts[0]
            word1 = word1_pos[:-2]
            word_pos1 = word1_pos[-1]


            rel = parts[3]
            word2_pos = parts[1]
            word2 = word2_pos[:-2]
            word_pos2 = word2_pos[-1]
            def1 = extract_word_deflist_given_POS(word1, word_pos1)
            if len(def1) ==0:
                def1 = extract_word_top_def_from_wordnet(word1)
            def2 = extract_word_deflist_given_POS(word2, word_pos2)
            if len(def2) ==0:
                def2 = extract_word_top_def_from_wordnet(word2)
            if len(def1)==0 or len(def2)==0:
                if len(def1)==0:
                    def1 = [word1]
                if len(def2)==0:
                    def2 = [word2]
                # remove_size+=1
                # print 'cut...', line
                # continue
            if rel == 'entailm':
                continue
            if rel == 'hyper':
                hyper_count+=1
            count+=1
            writefile.write(rel+'\t'+word1+'\t'+replace_punctuation_in_text_by_whitespace(def1[0])+'\t'+word2+'\t'+replace_punctuation_in_text_by_whitespace(def2[0])+'\n')
        readfile.close()
        writefile.close()
        print 'pair size:' , count,hyper_count,remove_size
    print 'all done'

def reformat_top_definitionData_to_subtasks():
    folder = '/save/wenpeng/datasets/UnsupervisedHypernymy/'
    files = ['BLESS.test.top.definition','EVALution.test.top.definition','LenciBenotto.test.top.definition','Weeds.test.top.definition']
    outfiles = ['BLESS.test.top.definition.hyper.vs.all','EVALution.test.top.definition.hyper.vs.all','LenciBenotto.test.top.definition.hyper.vs.all','Weeds.test.top.definition.hyper.vs.all']
    for i in range(len(files)):
        hyper_size = 0
        readfile =open(folder+files[i], 'r')
        writefile = open(folder+outfiles[i], 'w')
        for line in readfile:
            parts = line.strip().split('\t')
            label = parts[0]
            if label == 'hyper':
                writefile.write('1\t'+'\t'.join(parts[1:])+'\n')
                hyper_size+=1
            else:
                writefile.write('0\t'+'\t'.join(parts[1:])+'\n')
        writefile.close()
        readfile.close()
        print 'hyper_size:', hyper_size
    print 'hyper vs all over'

def load_from_raw_all_def_pairs():
    folder = '/save/wenpeng/datasets/EVALution/'
    files = ['EVALution.test']
    '''
    shadow-n        follow-n        True    hyper
    coffee-n        good-n  False   attrib
    space-v place-v False   synonym
    ceiling-n       floor-n False   antonym
    '''
    outfiles = ['EVALution.test.all.def.combinations.binary']
    count=0
    hyper_count = 0
    remove_size =0
    for i in range(len(files)):
        readfile = open(folder+files[i], 'r')
        writefile = open(folder+outfiles[i], 'w')
        for line in readfile:
            parts = line.strip().split()
            word1_pos = parts[0]
            word1 = word1_pos[:-2]
            # word_pos1 = word1_pos[-1]

            label = '0'
            rel = parts[3]
            word2_pos = parts[1]
            word2 = word2_pos[:-2]
            # word_pos2 = word2_pos[-1]
            def1_list = extract_word_all_def_list(word1)
            # if len(def1) ==0:
            #     def1 = extract_word_top_def_from_wordnet(word1)
            def2_list = extract_word_all_def_list(word2)
            # if len(def2) ==0:
            #     def2 = extract_word_top_def_from_wordnet(word2)
            if len(def1_list)==0 or len(def2_list)==0:
                remove_size+=1
                print 'cut...', line
                continue
            if rel == 'entailm':
                continue
            if rel == 'hyper':
                hyper_count+=1
                label = '1'
            count+=1
            for def1 in def1_list:
                for def2 in def2_list:
                    writefile.write(label+'\t'+word1+'\t'+def1+'\t'+word2+'\t'+def2+'\n')
        readfile.close()
        writefile.close()
    print 'pair size:' , count,hyper_count,remove_size

def load_from_raw_all_def_pairs_all4datasets():
    folder = '/save/wenpeng/datasets/UnsupervisedHypernymy/'
    files = ['BLESS.test','LenciBenotto.test','Weeds.test']
    '''
    shadow-n        follow-n        True    hyper
    coffee-n        good-n  False   attrib
    space-v place-v False   synonym
    ceiling-n       floor-n False   antonym
    '''
    outfiles = ['BLESS.test.all.def.combinations.binary','LenciBenotto.test.all.def.combinations.binary','Weeds.test.all.def.combinations.binary']

    for i in range(len(files)):
        count=0
        hyper_count = 0
        remove_size =0
        readfile = open(folder+files[i], 'r')
        writefile = open(folder+outfiles[i], 'w')
        for line in readfile:
            parts = line.strip().split()
            word1_pos = parts[0]
            word1 = word1_pos[:-2]
            # word_pos1 = word1_pos[-1]

            label = '0'
            rel = parts[3]
            word2_pos = parts[1]
            word2 = word2_pos[:-2]
            # word_pos2 = word2_pos[-1]
            def1_list = extract_word_all_def_list(word1)
            # if len(def1) ==0:
            #     def1 = extract_word_top_def_from_wordnet(word1)
            def2_list = extract_word_all_def_list(word2)
            # if len(def2) ==0:
            #     def2 = extract_word_top_def_from_wordnet(word2)
            # if len(def1_list)==0 or len(def2_list)==0:
            #     remove_size+=1
            #     print 'cut...', line
            #     continue
            if len(def1_list)==0 or len(def2_list)==0:
                if len(def1_list)==0:
                    def1_list = [word1]
                if len(def2_list)==0:
                    def2_list = [word2]
            if rel == 'entailm':
                continue
            if rel == 'hyper':
                hyper_count+=1
                label = '1'
            count+=1
            for def1 in def1_list:
                for def2 in def2_list:
                    writefile.write(label+'\t'+word1+'\t'+def1+'\t'+word2+'\t'+def2+'\n')
        readfile.close()
        writefile.close()
        print 'pair size:' , count,hyper_count,remove_size
    print 'all done'

if __name__ == '__main__':
    # Extract_Top_Def()
    reformat_top_definitionData_to_subtasks()
    # load_from_raw_all_def_pairs()
    # load_from_raw_all_def_pairs_all4datasets()
