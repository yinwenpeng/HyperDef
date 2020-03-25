
from common_functions import Conv_for_Pair
import theano.tensor as T

def four_way_attentiveConvNet(rng,batch_size, emb_size,hidden_size, filter_size,
embed_word1,embed_word2,word1_mask,word2_mask,max_term_len,
embed_input_l,sents_mask_l,embed_input_r,sents_mask_r,maxSentLen,
conv_W,conv_b,conv_W_context,conv_b_context):

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

    return p_ww, p_wd, p_dw, p_dd
