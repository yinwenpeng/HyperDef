import cPickle
import gzip
import os
import sys
sys.setrecursionlimit(6000)
import time

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import random

from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from theano.tensor.signal import downsample
from random import shuffle
from term_def_seek.extract_term_def_wildly import load_concept_def

from scipy.stats import mode


from load_data import load_word2id,parse_individual_termPair,load_SNLI_dataset, write_predictions_to_analysis,load_wordnet_hyper_vs_all_with_words,load_word2vec, load_word2vec_to_init, load_EVAlution_hyper_vs_all_with_words
from common_functions import load_model_from_file,apk, Conv_for_Pair,dropout_layer, store_model_to_file, elementwise_is_two,Conv_with_Mask_with_Gate, Conv_with_Mask, create_conv_para, L2norm_paraList, ABCNN, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
'''
1, word tokenized the definition sentences, not good
2, use maxpool and attentive_maxpool
'''

def evaluate_lenet5(term1_str, term2_str):
    emb_size=300
    filter_size=[3,3]
    maxSentLen=40
    hidden_size=[300,300]
    max_term_len=4
    p_mode = 'conc'
    batch_size = 1

    term1_def, source1 = load_concept_def(term1_str)
    print '\n',term1_str, ':\t', term1_def,'\t', source1,'\n'
    term2_def, source2 = load_concept_def(term2_str)
    print '\n',term2_str, ':\t', term2_def, '\t', source2,'\n'
    # exit(0)

    word2id = load_word2id('/save/wenpeng/datasets/HypeNet/HyperDef_label_meta_best_para_word2id.pkl')
    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results

    # all_sentences_l, all_masks_l, all_sentences_r, all_masks_r, all_word1,all_word2,all_word1_mask,all_word2_mask,all_labels, all_extra, word2id  =load_wordnet_hyper_vs_all_with_words(maxlen=maxSentLen, wordlen=max_term_len)  #minlen, include one label, at least one word in the sentence
    # test_sents_l, test_masks_l, test_sents_r, test_masks_r, test_labels, word2id  =load_ACE05_dataset(maxSentLen, word2id)
    # test_sents_l, test_masks_l, test_sents_r, test_masks_r, test_word1,test_word2,test_word1_mask,test_word2_mask,test_labels, test_extra, word2id = load_EVAlution_hyper_vs_all_with_words(maxSentLen, word2id, wordlen=max_term_len)
    test_sents_l, test_masks_l, test_sents_r, test_masks_r, test_word1,test_word2,test_word1_mask,test_word2_mask, test_extra, word2id = parse_individual_termPair(term1_str, term2_str, term1_def, term2_def, maxSentLen, word2id, wordlen=max_term_len)
    # total_size = len(all_sentences_l)
    # hold_test_size = 10000
    # train_size = total_size - hold_test_size



    # train_sents_l=np.asarray(all_sentences_l[:train_size], dtype='int32')
    # dev_sents_l=np.asarray(all_sentences_l[1], dtype='int32')
    # test_sents_l=np.asarray(all_sentences_l[-test_size:], dtype='int32')
    test_sents_l=np.asarray(test_sents_l, dtype='int32')

    # train_masks_l=np.asarray(all_masks_l[:train_size], dtype=theano.config.floatX)
    # dev_masks_l=np.asarray(all_masks_l[1], dtype=theano.config.floatX)
    # test_masks_l=np.asarray(all_masks_l[-test_size:], dtype=theano.config.floatX)
    test_masks_l=np.asarray(test_masks_l, dtype=theano.config.floatX)

    # train_sents_r=np.asarray(all_sentences_r[:train_size], dtype='int32')
    # dev_sents_r=np.asarray(all_sentences_r[1]    , dtype='int32')
    # test_sents_r=np.asarray(all_sentences_r[-test_size:], dtype='int32')
    test_sents_r=np.asarray(test_sents_r, dtype='int32')

    # train_masks_r=np.asarray(all_masks_r[:train_size], dtype=theano.config.floatX)
    # dev_masks_r=np.asarray(all_masks_r[1], dtype=theano.config.floatX)
    # test_masks_r=np.asarray(all_masks_r[-test_size:], dtype=theano.config.floatX)
    test_masks_r=np.asarray(test_masks_r, dtype=theano.config.floatX)

    # train_word1=np.asarray(all_word1[:train_size], dtype='int32')
    # train_word2=np.asarray(all_word2[:train_size], dtype='int32')
    test_word1=np.asarray(test_word1, dtype='int32')
    test_word2=np.asarray(test_word2, dtype='int32')

    # train_word1_mask=np.asarray(all_word1_mask[:train_size], dtype=theano.config.floatX)
    # train_word2_mask=np.asarray(all_word2_mask[:train_size], dtype=theano.config.floatX)
    test_word1_mask=np.asarray(test_word1_mask, dtype=theano.config.floatX)
    test_word2_mask=np.asarray(test_word2_mask, dtype=theano.config.floatX)

    # train_labels_store=np.asarray(all_labels[:train_size], dtype='int32')
    # dev_labels_store=np.asarray(all_labels[1], dtype='int32')
    # test_labels_store=np.asarray(all_labels[-test_size:], dtype='int32')
    # test_labels_store=np.asarray(test_labels, dtype='int32')

    # train_extra=np.asarray(all_extra[:train_size], dtype=theano.config.floatX)
    test_extra=np.asarray(test_extra, dtype=theano.config.floatX)

    # train_size=len(train_labels_store)
    # dev_size=len(dev_labels_store)
    test_size=len(test_extra)
    print ' test size: ', len(test_extra)

    vocab_size=len(word2id)+1


    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
    # rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    # id2word = {y:x for x,y in word2id.iteritems()}
    # word2vec=load_word2vec()
    # rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    init_embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable
    # store_model_to_file('/save/wenpeng/datasets/HypeNet/HyperDef_label_meta_best_para_embeddings', [init_embeddings])
    # exit(0)
    #now, start to build the input form of the model
    sents_ids_l=T.imatrix()
    sents_mask_l=T.fmatrix()
    sents_ids_r=T.imatrix()
    sents_mask_r=T.fmatrix()
    word1_ids = T.imatrix()
    word2_ids = T.imatrix()
    word1_mask = T.fmatrix()
    word2_mask = T.fmatrix()
    extra = T.fvector()
    # labels=T.ivector()
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    def embed_input(emb_matrix, sent_ids):
        return emb_matrix[sent_ids.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)

    embed_input_l=embed_input(init_embeddings, sents_ids_l)#embeddings[sents_ids_l.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1) #the input format can be adapted into CNN or GRU or LSTM
    embed_input_r=embed_input(init_embeddings, sents_ids_r)#embeddings[sents_ids_r.flatten()].reshape((batch_size,maxSentLen, emb_size)).dimshuffle(0,2,1)

    embed_word1 = init_embeddings[word1_ids.flatten()].reshape((batch_size,word1_ids.shape[1], emb_size))
    embed_word2 = init_embeddings[word2_ids.flatten()].reshape((batch_size,word2_ids.shape[1], emb_size))
    word1_embedding = T.sum(embed_word1*word1_mask.dimshuffle(0,1,'x'), axis=1)
    word2_embedding = T.sum(embed_word2*word2_mask.dimshuffle(0,1,'x'), axis=1)


    '''create_AttentiveConv_params '''
    conv_W, conv_b=create_conv_para(rng, filter_shape=(hidden_size[1], 1, emb_size, filter_size[0]))
    conv_W_context, conv_b_context=create_conv_para(rng, filter_shape=(hidden_size[1], 1, emb_size, 1))

    NN_para=[conv_W, conv_b,conv_W_context]

    '''
    attentive convolution function
    '''
    term_vs_term_layer = Conv_for_Pair(rng,
            origin_input_tensor3=embed_word1.dimshuffle(0,2,1),
            origin_input_tensor3_r = embed_word2.dimshuffle(0,2,1),
            input_tensor3=embed_word1.dimshuffle(0,2,1),
            input_tensor3_r = embed_word2.dimshuffle(0,2,1),
             mask_matrix = word1_mask,
             mask_matrix_r = word2_mask,
             image_shape=(batch_size, 1, emb_size, max_term_len),
             image_shape_r = (batch_size, 1, emb_size, max_term_len),
             filter_shape=(hidden_size[1], 1, emb_size, filter_size[0]),
             filter_shape_context=(hidden_size[1], 1,emb_size, 1),
             W=conv_W, b=conv_b,
             W_context=conv_W_context, b_context=conv_b_context)
    tt_embeddings_l = term_vs_term_layer.attentive_maxpool_vec_l
    tt_embeddings_r = term_vs_term_layer.attentive_maxpool_vec_r

    p_ww = T.concatenate([tt_embeddings_l,tt_embeddings_r,tt_embeddings_l*tt_embeddings_r,tt_embeddings_l-tt_embeddings_r], axis=1)

    term_vs_def_layer = Conv_for_Pair(rng,
            origin_input_tensor3=embed_word1.dimshuffle(0,2,1),
            origin_input_tensor3_r = embed_input_r,
            input_tensor3=embed_word1.dimshuffle(0,2,1),
            input_tensor3_r = embed_input_r,
             mask_matrix = word1_mask,
             mask_matrix_r = sents_mask_r,
             image_shape=(batch_size, 1, emb_size, max_term_len),
             image_shape_r = (batch_size, 1, emb_size, maxSentLen),
             filter_shape=(hidden_size[1], 1, emb_size, filter_size[0]),
             filter_shape_context=(hidden_size[1], 1,emb_size, 1),
             W=conv_W, b=conv_b,
             W_context=conv_W_context, b_context=conv_b_context)
    td_embeddings_l = term_vs_def_layer.attentive_maxpool_vec_l
    td_embeddings_r = term_vs_def_layer.attentive_maxpool_vec_r
    p_wd = T.concatenate([td_embeddings_l,td_embeddings_r,td_embeddings_l*td_embeddings_r,td_embeddings_l-td_embeddings_r], axis=1)


    def_vs_term_layer = Conv_for_Pair(rng,
            origin_input_tensor3=embed_input_l,
            origin_input_tensor3_r = embed_word2.dimshuffle(0,2,1),
            input_tensor3=embed_input_l,
            input_tensor3_r = embed_word2.dimshuffle(0,2,1),
             mask_matrix = sents_mask_l,
             mask_matrix_r = word2_mask,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             image_shape_r = (batch_size, 1, emb_size, max_term_len),
             filter_shape=(hidden_size[1], 1, emb_size, filter_size[0]),
             filter_shape_context=(hidden_size[1], 1,emb_size, 1),
             W=conv_W, b=conv_b,
             W_context=conv_W_context, b_context=conv_b_context)
    dt_embeddings_l = def_vs_term_layer.attentive_maxpool_vec_l
    dt_embeddings_r = def_vs_term_layer.attentive_maxpool_vec_r

    p_dw = T.concatenate([dt_embeddings_l,dt_embeddings_r,dt_embeddings_l*dt_embeddings_r,dt_embeddings_l-dt_embeddings_r], axis=1)


    def_vs_def_layer = Conv_for_Pair(rng,
            origin_input_tensor3=embed_input_l,
            origin_input_tensor3_r = embed_input_r,
            input_tensor3=embed_input_l,
            input_tensor3_r = embed_input_r,
             mask_matrix = sents_mask_l,
             mask_matrix_r = sents_mask_r,
             image_shape=(batch_size, 1, emb_size, maxSentLen),
             image_shape_r = (batch_size, 1, emb_size, maxSentLen),
             filter_shape=(hidden_size[1], 1, emb_size, filter_size[0]),
             filter_shape_context=(hidden_size[1], 1,emb_size, 1),
             W=conv_W, b=conv_b,
             W_context=conv_W_context, b_context=conv_b_context)
    dd_embeddings_l = def_vs_def_layer.attentive_maxpool_vec_l
    dd_embeddings_r = def_vs_def_layer.attentive_maxpool_vec_r
    p_dd = T.concatenate([dd_embeddings_l,dd_embeddings_r,dd_embeddings_l*dd_embeddings_r,dd_embeddings_l-dd_embeddings_r], axis=1)

    if p_mode == 'conc':
        p=T.concatenate([p_ww, p_wd, p_dw, p_dd], axis=1)
        p_len = 4*4*hidden_size[1]
    else:
        p = T.max(T.concatenate([p_ww.dimshuffle('x',0,1),p_wd.dimshuffle('x',0,1),p_dw.dimshuffle('x',0,1),p_dd.dimshuffle('x',0,1)],axis=0), axis=0)
        p_len =4*hidden_size[1]
    "form input to LR classifier"
    LR_input = T.concatenate([p,extra.dimshuffle(0,'x')],axis=1)
    LR_input_size=p_len+1

    U_a = create_ensemble_para(rng, 2, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((2,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a, LR_b]


    layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=2, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector



    params = NN_para+LR_para #[init_embeddings]
    load_model_from_file('/save/wenpeng/datasets/HypeNet/HyperDef_label_meta_best_para_embeddings', [init_embeddings])

    load_model_from_file('/save/wenpeng/datasets/HypeNet/HyperDef_label_meta_best_para_0.938730853392', params)

    test_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, word1_ids,word2_ids,word1_mask,word2_mask,extra], [layer_LR.y_pred,layer_LR.prop_for_posi], allow_input_downcast=True, on_unused_input='ignore')

    ###############
    # TRAIN MODEL #
    ###############
    print '... testing'


    n_test_batches=test_size/batch_size
    n_test_remain = test_size%batch_size
    if n_test_remain!=0:
        test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]
    else:
        test_batch_start=list(np.arange(n_test_batches)*batch_size)



    # max_acc_dev=0.0
    # max_ap_test=0.0
    # max_ap_topk_test=0.0
    # max_f1=0.0

    # cost_i=0.0
    # train_indices = range(train_size)


    for idd, test_batch_id in enumerate(test_batch_start): # for each test batch
        pred_i, prob_i=test_model(
                test_sents_l[test_batch_id:test_batch_id+batch_size],
                test_masks_l[test_batch_id:test_batch_id+batch_size],
                test_sents_r[test_batch_id:test_batch_id+batch_size],
                test_masks_r[test_batch_id:test_batch_id+batch_size],
                test_word1[test_batch_id:test_batch_id+batch_size],
                test_word2[test_batch_id:test_batch_id+batch_size],
                test_word1_mask[test_batch_id:test_batch_id+batch_size],
                test_word2_mask[test_batch_id:test_batch_id+batch_size],
                test_extra[test_batch_id:test_batch_id+batch_size])
        print pred_i, prob_i





if __name__ == '__main__':
    evaluate_lenet5(sys.argv[1], sys.argv[2])
