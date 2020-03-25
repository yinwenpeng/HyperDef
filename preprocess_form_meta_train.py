import codecs


def form_meta_data():
    size = 0
    pos_size = 0
    writefile = codecs.open('/save/wenpeng/datasets/wordnet-related/meta_data_hyper_or_not.txt', 'w', 'utf-8')

    readfile = codecs.open('/save/wenpeng/datasets/wordnet-related/word-hyper-vs-all-PairPerLine.txt', 'r', 'utf-8')
    for line in readfile:

        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==6:

            def_l=parts[2].strip()
            def_r=parts[4].strip()

            word1 = ' '.join(parts[1][:parts[1].find('.')].split('_'))
            word2 = ' '.join(parts[3][:parts[3].find('.')].split('_'))
            label = parts[0].strip()
            if label == '1':
                pos_size+=1
            writefile.write(label+'\t'+word1+'\t'+def_l+'\t'+word2+'\t'+def_r+'\n')
            size+=1
    readfile.close()
    print 'size: ', size
    readfile = codecs.open('/save/wenpeng/datasets/HypeNet/dataset_lex/train.as.input.txt', 'r', 'utf-8')
    for line in readfile:

        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==5:
            def_l=parts[2].strip()
            def_r=parts[4].strip()

            word1 = parts[1].strip()
            word2 = parts[3].strip()

            label = parts[0].strip()
            if label == '1':
                pos_size+=1
            writefile.write(label+'\t'+word1+'\t'+def_l+'\t'+word2+'\t'+def_r+'\n')
            size+=1
    readfile.close()
    print 'size: ', size
    readfile = codecs.open('/save/wenpeng/datasets/HypeNet/dataset_lex/test.as.input.txt', 'r', 'utf-8')
    for line in readfile:

        parts=line.strip().split('\t') #lowercase all tokens, as we guess this is not important for sentiment task
        if len(parts)==5:
            def_l=parts[2].strip()
            def_r=parts[4].strip()

            word1 = parts[1].strip()
            word2 = parts[3].strip()

            label = parts[0].strip()
            if label == '1':
                pos_size+=1
            writefile.write(label+'\t'+word1+'\t'+def_l+'\t'+word2+'\t'+def_r+'\n')
            size+=1
    readfile.close()
    print 'size: ', size, ' pos_size: ', pos_size, ' neg_size: ', size-pos_size
    writefile.close()


if __name__ == '__main__':
    form_meta_data()
