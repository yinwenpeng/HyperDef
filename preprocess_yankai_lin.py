


def reformat():
    co=0
    files = ['/save/wenpeng/datasets/Yankai_Lin_ACL2016_ER/test.txt']#,'/save/wenpeng/datasets/Yankai_Lin_ACL2016_ER/test.txt']
    rel_set = set()
    for fil in files:
        readfile = open(fil, 'r')
        for line in readfile:
            parts = line.strip().split('\t')
            rel_set.add(parts[4])
            # if parts[-1] != '###END###':
            #     print line
            #     print 'parts[-1]:', parts[-1]
            #     co+=1

        readfile.close()
    print 'co:', co
    print len(rel_set)
    for rel in rel_set:
        print rel

    # readfile = open('/save/wenpeng/datasets/Yankai_Lin_ACL2016_ER/relation2id.txt', 'r')
    # rels = set()
    # for line in readfile:
    #     rels.add(line.strip().split()[0])
    # readfile.close()
    # print rel_set - rels


if __name__ == '__main__':
    reformat()
