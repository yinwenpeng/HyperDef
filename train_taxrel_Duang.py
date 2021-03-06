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


from scipy.stats import mode


from load_data import store_word2id, load_SNLI_dataset, write_predictions_to_analysis,load_TaxRel_DuangDo,load_word2vec, load_word2vec_to_init, load_EVAlution_hyper_vs_all_with_words
from common_functions import apk, Conv_for_Pair,dropout_layer, store_model_to_file, elementwise_is_two,Conv_with_Mask_with_Gate, Conv_with_Mask, create_conv_para, L2norm_paraList, ABCNN, create_ensemble_para, cosine_matrix1_matrix2_rowwise, Diversify_Reg, Gradient_Cost_Para, GRU_Batch_Tensor_Input_with_Mask, create_LSTM_para
'''
1, word tokenized the definition sentences, not good
2, use maxpool and attentive_maxpool
'''

def evaluate_lenet5(learning_rate=0.02, n_epochs=4, L2_weight=1e-5, extra_size=4, emb_size=300, batch_size=50, filter_size=[3,3], maxSentLen=40, hidden_size=[300,300], max_term_len=4, p_mode = 'conc'):

    model_options = locals().copy()
    print "model options", model_options

    seed=1234
    np.random.seed(seed)
    rng = np.random.RandomState(seed)    #random seed, control the model generates the same results


    all_sentences_l, all_masks_l, all_sentences_r, all_masks_r, all_word1,all_word2,all_word1_mask,all_word2_mask,all_labels, all_extra, word2id  =load_TaxRel_DuangDo(maxlen=maxSentLen, wordlen=max_term_len, word2id=None,filname='train')  #minlen, include one label, at least one word in the sentence
    # test_sents_l, test_masks_l, test_sents_r, test_masks_r, test_labels, word2id  =load_ACE05_dataset(maxSentLen, word2id)
    test_sents_l, test_masks_l, test_sents_r, test_masks_r, test_word1,test_word2,test_word1_mask,test_word2_mask,test_labels, test_extra, word2id = load_TaxRel_DuangDo(maxlen=maxSentLen, wordlen=max_term_len, word2id=word2id,filname='test')
    # store_word2id(word2id, '/save/wenpeng/datasets/EVALution/HyperDef_label_4ways_conc_best_para_word2id.pkl')
    # exit(0)

    train_size = len(all_sentences_l)



    train_sents_l=np.asarray(all_sentences_l, dtype='int32')
    # dev_sents_l=np.asarray(all_sentences_l[1], dtype='int32')
    # test_sents_l=np.asarray(all_sentences_l[-test_size:], dtype='int32')
    test_sents_l=np.asarray(test_sents_l, dtype='int32')

    train_masks_l=np.asarray(all_masks_l, dtype=theano.config.floatX)
    # dev_masks_l=np.asarray(all_masks_l[1], dtype=theano.config.floatX)
    # test_masks_l=np.asarray(all_masks_l[-test_size:], dtype=theano.config.floatX)
    test_masks_l=np.asarray(test_masks_l, dtype=theano.config.floatX)

    train_sents_r=np.asarray(all_sentences_r, dtype='int32')
    # dev_sents_r=np.asarray(all_sentences_r[1]    , dtype='int32')
    # test_sents_r=np.asarray(all_sentences_r[-test_size:], dtype='int32')
    test_sents_r=np.asarray(test_sents_r, dtype='int32')

    train_masks_r=np.asarray(all_masks_r, dtype=theano.config.floatX)
    # dev_masks_r=np.asarray(all_masks_r[1], dtype=theano.config.floatX)
    # test_masks_r=np.asarray(all_masks_r[-test_size:], dtype=theano.config.floatX)
    test_masks_r=np.asarray(test_masks_r, dtype=theano.config.floatX)

    train_word1=np.asarray(all_word1, dtype='int32')
    train_word2=np.asarray(all_word2, dtype='int32')
    test_word1=np.asarray(test_word1, dtype='int32')
    test_word2=np.asarray(test_word2, dtype='int32')

    train_word1_mask=np.asarray(all_word1_mask, dtype=theano.config.floatX)
    train_word2_mask=np.asarray(all_word2_mask, dtype=theano.config.floatX)
    test_word1_mask=np.asarray(test_word1_mask, dtype=theano.config.floatX)
    test_word2_mask=np.asarray(test_word2_mask, dtype=theano.config.floatX)

    train_labels_store=np.asarray(all_labels, dtype='int32')
    # dev_labels_store=np.asarray(all_labels[1], dtype='int32')
    # test_labels_store=np.asarray(all_labels[-test_size:], dtype='int32')
    test_labels_store=np.asarray(test_labels, dtype='int32')

    train_extra=np.asarray(all_extra, dtype=theano.config.floatX)
    test_extra=np.asarray(test_extra, dtype=theano.config.floatX)

    # train_size=len(train_labels_store)
    # dev_size=len(dev_labels_store)
    test_size=len(test_labels_store)
    print 'train size: ', train_size,  ' test size: ', test_size

    vocab_size=len(word2id)+1
    print 'vocab size:', vocab_size-1


    rand_values=rng.normal(0.0, 0.01, (vocab_size, emb_size))   #generate a matrix by Gaussian distribution
    #here, we leave code for loading word2vec to initialize words
    rand_values[0]=np.array(np.zeros(emb_size),dtype=theano.config.floatX)
    id2word = {y:x for x,y in word2id.iteritems()}
    word2vec=load_word2vec()
    rand_values=load_word2vec_to_init(rand_values, id2word, word2vec)
    init_embeddings=theano.shared(value=np.array(rand_values,dtype=theano.config.floatX), borrow=True)   #wrap up the python variable "rand_values" into theano variable


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
    labels=T.ivector()
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

    # HL_input = T.concatenate([p,cosine_matrix1_matrix2_rowwise(word1_embedding,word2_embedding).dimshuffle(0,'x'),extra.dimshuffle(0,'x')],axis=1)
    # HL_input_size=p_len+1+1
    #
    # HL_layer_1=HiddenLayer(rng, input=HL_input, n_in=HL_input_size, n_out=hidden_size[1], activation=T.tanh)


    "form input to LR classifier"
    LR_input = T.concatenate([p,cosine_matrix1_matrix2_rowwise(word1_embedding,word2_embedding).dimshuffle(0,'x'),extra.dimshuffle(0,'x')],axis=1)
    LR_input_size=p_len+1+1
    # LR_input = HL_layer_1.output
    # LR_input_size = hidden_size[1]

    U_a = create_ensemble_para(rng, 4, LR_input_size) # the weight matrix hidden_size*2
    LR_b = theano.shared(value=np.zeros((4,),dtype=theano.config.floatX),name='LR_b', borrow=True)  #bias for each target class
    LR_para=[U_a, LR_b]


    layer_LR=LogisticRegression(rng, input=LR_input, n_in=LR_input_size, n_out=4, W=U_a, b=LR_b) #basically it is a multiplication between weight matrix and input feature vector
    loss=layer_LR.negative_log_likelihood(labels)  #for classification task, we usually used negative log likelihood as loss, the lower the better.


    # L2_reg = (conv_W**2).sum()+(conv_W_context**2).sum()+(U_a**2).sum()





    params = NN_para+LR_para #[init_embeddings]

    cost=loss#+L2_weight*L2_reg

    updates =   Gradient_Cost_Para(cost,params, learning_rate)


    train_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, word1_ids,word2_ids,word1_mask,word2_mask,extra,labels], cost, updates=updates, allow_input_downcast=True, on_unused_input='ignore')
    test_model = theano.function([sents_ids_l, sents_mask_l, sents_ids_r, sents_mask_r, word1_ids,word2_ids,word1_mask,word2_mask,extra,labels], layer_LR.errors(labels), allow_input_downcast=True, on_unused_input='ignore')

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 50000000000  # look as this many examples regardless
    start_time = time.time()
    mid_time = start_time
    past_time= mid_time
    epoch = 0
    done_looping = False

    n_train_batches=train_size/batch_size
    train_batch_start=list(np.arange(n_train_batches)*batch_size)+[train_size-batch_size]
    # n_dev_batches=dev_size/batch_size
    # dev_batch_start=list(np.arange(n_dev_batches)*batch_size)+[dev_size-batch_size]
    n_test_batches=test_size/batch_size
    n_test_remain = test_size%batch_size
    if n_test_remain!=0:
        test_batch_start=list(np.arange(n_test_batches)*batch_size)+[test_size-batch_size]
    else:
        test_batch_start=list(np.arange(n_test_batches)*batch_size)



    # max_acc_dev=0.0
    max_acc_test=0.0


    cost_i=0.0
    train_indices = range(train_size)

    while epoch < n_epochs:
        epoch = epoch + 1

        random.Random(100).shuffle(train_indices) #shuffle training set for each new epoch, is supposed to promote performance, but not garrenteed

        iter_accu=0

        for batch_id in train_batch_start: #for each batch
            # iter means how many batches have been run, taking into loop
            iter = (epoch - 1) * n_train_batches + iter_accu +1
            iter_accu+=1
            train_id_batch = train_indices[batch_id:batch_id+batch_size]
            cost_i+= train_model(
                                train_sents_l[train_id_batch],
                                train_masks_l[train_id_batch],
                                train_sents_r[train_id_batch],
                                train_masks_r[train_id_batch],
                                train_word1[train_id_batch],
                                train_word2[train_id_batch],
                                train_word1_mask[train_id_batch],
                                train_word2_mask[train_id_batch],
                                train_extra[train_id_batch],
                                train_labels_store[train_id_batch])

            #after each 1000 batches, we test the performance of the model on all test data
            if iter%10==0:
                print 'Epoch ', epoch, 'iter '+str(iter)+' average cost: '+str(cost_i/iter), 'uses ', (time.time()-past_time)/60.0, 'min'
                past_time = time.time()

                pred_labels =[]
                probs = []
                gold_labels =[]
                error_sum=0.0
                for idd, test_batch_id in enumerate(test_batch_start): # for each test batch
                    error_i=test_model(
                            test_sents_l[test_batch_id:test_batch_id+batch_size],
                            test_masks_l[test_batch_id:test_batch_id+batch_size],
                            test_sents_r[test_batch_id:test_batch_id+batch_size],
                            test_masks_r[test_batch_id:test_batch_id+batch_size],
                            test_word1[test_batch_id:test_batch_id+batch_size],
                            test_word2[test_batch_id:test_batch_id+batch_size],
                            test_word1_mask[test_batch_id:test_batch_id+batch_size],
                            test_word2_mask[test_batch_id:test_batch_id+batch_size],
                            test_extra[test_batch_id:test_batch_id+batch_size],
                            test_labels_store[test_batch_id:test_batch_id+batch_size])

                    error_sum+=error_i
                    # pred_labels+=list(pred_i)
                    # probs+=list(prob_i)
                # if n_test_remain !=0:
                #     probs = probs[:(len(test_batch_start)-1)*batch_size]+probs[-n_test_remain:]
                # assert len(test_labels) == len(probs)
                test_acc=1.0-error_sum/(len(test_batch_start))
                # test_ap = apk(test_labels, probs, k=len(test_labels))
                # test_ap_top100 = apk(test_labels, probs, k=100)
                if test_acc>max_acc_test:
                    max_acc_test = test_acc


                print '\t\tcurrent acc:', test_acc,' ; ','\tmax_acc_test: ', max_acc_test



        print 'Epoch ', epoch, 'uses ', (time.time()-mid_time)/60.0, 'min'
        mid_time = time.time()

        #print 'Batch_size: ', update_freq
    end_time = time.time()

    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))





if __name__ == '__main__':
    evaluate_lenet5()
    #def evaluate_lenet5(learning_rate=0.01, n_epochs=4, L2_weight=0.0000001, div_weight=0.00001, emb_size=300, batch_size=50, filter_size=[3,1], maxSentLen=50, hidden_size=[300,300], margin =0.2, comment='HL relu'):

#     lr_list=[0.01,0.02,0.008,0.005]
#     batch_list=[3,5,10,20,30,40,50,60,70,80,100]
#     maxlen_list=[35,40,45,50,55,60,65,70,75,80]
#
#     best_acc=0.0
#     best_lr=0.01
#     for lr in lr_list:
#         acc_test= evaluate_lenet5(learning_rate=lr)
#         if acc_test>best_acc:
#             best_lr=lr
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#     best_batch=50
#     for batch in batch_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr,  batch_size=batch)
#         if acc_test>best_acc:
#             best_batch=batch
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#
#     best_maxlen=50
#     for maxlen in maxlen_list:
#         acc_test= evaluate_lenet5(learning_rate=best_lr,  batch_size=best_batch, maxSentLen=maxlen)
#         if acc_test>best_acc:
#             best_maxlen=maxlen
#             best_acc=acc_test
#         print '\t\t\t\tcurrent best_acc:', best_acc
#     print 'Hyper tune finished, best test acc: ', best_acc, ' by  lr: ', best_lr, ' batch: ', best_batch, ' maxlen: ', best_maxlen
