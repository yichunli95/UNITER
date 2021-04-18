"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

MLM datasets
"""
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import (mlm_DetectFeatTxtTokDataset, TxtTokLmdb,
                   pad_tensors, get_gather_index)
from pytorch_pretrained_bert import BertTokenizer
from cytoolz import curry
import spacy
from sprl import * 
from spacy.tokenizer import Tokenizer
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import word_tokenize
nlp = spacy.load('/content/UNITER/models/en_core_web_lg-sprl')
device = torch.device("cuda")


@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids

tokenizer1 = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)
tokenizer2 = bert_tokenize(tokenizer1)

def mask_spatial(example, vocab_range, mask):
    input_ids = []
    output_label = []
    # 1. spacy tokenize the sentence and sprl-spacy find the spatial words
    old_tokens = nlp(example['sentence'])
    old_tokens = [t.text for t in old_tokens]
    relations = sprl(example['sentence'], nlp, model_relext_filename='models/model_svm_relations.pkl')

    # 2. replace the spatial tokens with mask only if bert tokenize it as one word
    mask_to_old_bert_token = {}
    for rel in relations:
        start, end = rel[1].start, rel[1].end
        all_single = True
        for i in range(start, end):
            bert_token = tokenizer1.tokenize(old_tokens[i])
            tid = tokenizer1.convert_tokens_to_ids(bert_token)
            if len(tid) == 1:
                mask_to_old_bert_token[i] = tid[0]
                #old_tokens[i] = '[MASK]'
            else:
                all_single = False
                break
        if all_single:
            for i in range(start, end):
                old_tokens[i] = '[MASK]'

    # 3. use bert to tokenize and generate input_ids and output_label
    for i, token in enumerate(old_tokens):
        if token != '[MASK]':
            wd = tokenizer1.tokenize(token)
            ids = tokenizer1.convert_tokens_to_ids(wd)
            output_label.extend([-1]*len(ids))
            input_ids.extend(ids)
        else:
            input_ids.append(mask)
            output_label.append(mask_to_old_bert_token[i])

    if all(o == -1 for o in output_label):
        # at least mask 1
        #output_label[0] = example['input_ids'][0]
        output_label[0] = tokenizer1.convert_tokens_to_ids(tokenizer1.tokenize('.'))[0]
        input_ids[0] = mask
    
    assert len(input_ids) == len(output_label)
    return input_ids, output_label


def random_word(example, vocab_range, mask):
    """
    Masking some random prepositional tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []
    old_tokens = word_tokenize(example['sentence'])    
    input_ids = []
    is_subset = False
    i = 0
    while i < len(old_tokens):
        word = old_tokens[i].lower()
        two, three = None, None
        if i + 1 < len(old_tokens):
            two = ' '.join([word, old_tokens[i+1].lower()])
        if i + 2 < len(old_tokens):
            three = ' '.join([word, old_tokens[i+1].lower(), old_tokens[i+2].lower()])
        if word in prepositions:
            output_label, input_ids = random_replace(1, old_tokens, i, mask, vocab_range, output_label, input_ids)
            i += 1
        elif two in prepositions:
            output_label, input_ids = random_replace(2, old_tokens, i, mask, vocab_range, output_label, input_ids)
            i += 2
        elif three in prepositions:
            output_label, input_ids = random_replace(3, old_tokens, i, mask, vocab_range, output_label, input_ids)
            i += 3
        else:
            wd = tokenizer1.tokenize(word)
            ids = tokenizer1.convert_tokens_to_ids(wd)
            output_label.extend([-1]*len(ids))
            input_ids.extend(ids)
            i += 1
    
    example['input_ids'] = input_ids
    # print("Mask example['sent']:", example['sentence'])
    # print("Mask example['input_ids']:", example['input_ids'])

    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = example['input_ids'][0]
        input_ids[0] = mask

    # print(f'len(input_ids) is {len(input_ids)}')
    # print(f'len(output_label) is {len(output_label)}')
    # input_ids, txt_labels
    return input_ids, output_label

def random_replace(num_token, token_list, i, mask, vocab_range, output_label, input_ids):
    for ct in range(i, i + num_token):
        wd = tokenizer1.tokenize(token_list[ct])
        tid = tokenizer1.convert_tokens_to_ids(wd)
        if len(tid) == 1:
            token_list[ct] = '[MASK]'
            input_ids.append(mask)
            output_label.append(tid[0])
        else:
            output_label.extend(tid)
            input_ids.extend(tid)
    return output_label, input_ids

class SpatialMlmDataset(mlm_DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db):
        assert isinstance(txt_db, TxtTokLmdb)
        super().__init__(txt_db, img_db)
        
    def __getitem__(self, i):
        """
        Return:
        - input_ids    : (L, ), i.e., [cls, wd, wd, ..., sep, 0, 0], 0s padded
        - img_feat     : (num_bb, d)
        - img_pos_feat : (num_bb, 7)
        - attn_masks   : (L + num_bb, ), ie., [1, 1, ..., 0, 0, 1, 1]
        - txt_labels   : (L, ), [-1, -1, wid, -1, -1, -1]
        0's padded so that (L + num_bb) % 8 == 0
        """
        example = super().__getitem__(i)

        # text input
        input_ids, txt_labels = self.create_mlm_io(example)

        # img input
        img_feat, img_pos_feat, num_bb = self.mlm_get_img_feat(
            example['img_fname'])

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids.to(device), img_feat.to(device), img_pos_feat.to(device), attn_masks.to(device), txt_labels.to(device)

    def create_mlm_io(self, example):
        input_ids, txt_labels = mask_spatial(example,
                                            self.txt_db.v_range,
                                            self.txt_db.mask)
        input_ids = torch.tensor([self.txt_db.cls_]
                                 + input_ids
                                 + [self.txt_db.sep])
        txt_labels = torch.tensor([-1] + txt_labels + [-1])
        return input_ids, txt_labels
    
    def mlm_get_img_feat(self, fname_list):
        img_feats = []
        img_pos_feats = []
        num_bb = 0
        for i, img in enumerate(fname_list):
            feat, pos, nbb = self._get_img_feat(img)
            img_feats.append(feat)
            img_pos_feats.append(pos)
            num_bb += nbb
        img_feat = torch.cat(img_feats, dim=0)
        img_pos_feat = torch.cat(img_pos_feats, dim=0)
        return img_feat.to(device), img_pos_feat.to(device), num_bb

def spatial_mlm_collate(inputs):
    """
    Return:
    :input_ids    (n, max_L) padded with 0
    :position_ids (n, max_L) padded with 0
    :txt_lens     list of [txt_len]
    :img_feat     (n, max_num_bb, feat_dim)
    :img_pos_feat (n, max_num_bb, 7)
    :num_bbs      list of [num_bb]
    :attn_masks   (n, max_{L + num_bb}) padded with 0
    :txt_labels   (n, max_L) padded with -1
    """
    (input_ids, img_feats, img_pos_feats, attn_masks, txt_labels
     ) = map(list, unzip(inputs))

    # text batches
    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).to(device)
    txt_labels = pad_sequence(txt_labels, batch_first=True, padding_value=-1).to(device)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0).to(device)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs).to(device)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs).to(device)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0).to(device)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'txt_labels': txt_labels}
    return batch

prepositions = [
        "aboard",
        "about",
        "above",
        "absent",
        "across",
        "after",
        "against",
        "along",
        "alongside",
        "amid",
        "amidst",
        "among",
        "amongst",
        "around",
        "as",
        "astride",
        "at",
        "atop",
        "before",
        "afore",
        "behind",
        "below",
        "beneath",
        "beside",
        "besides",
        "between",
        #"beyond",
        "by",
        "circa",
        #"despite",
        "down",
        #"during",
        #"except",
        "for",
        "from",
        "in",
        "inside",
        "into",
        #"less",
        #"like",
        #"minus",
        "near",
        "nearer",
        "nearest",
        #"notwithstanding",
        #"of",
        "off",
        "on",
        "onto",
        "opposite",
        "outside",
        "over",
        "past",
        "per",
        "save",
        "since",
        "through",
        #"throughout",
        #"to",
        "toward",
        "towards",
        "under",
        "underneath",
        #"until",
        "up",
        "upon",
        "upside",
        #"versus",
        #"via",
        "with",
        "within",
        #"without",
        #"worth",
        #"according to",
        "adjacent to",
        "ahead of",
        "apart from",
        #"as of",
        #"as per",
        "as regards",
        "aside from",
        "astern of",
        "back to",
        #"because of",
        "close to",
        #"due to",
        #"except for",
        "far from",
        "inside of",
        #"instead of",
        "left of",
        "near to",
        "next to",
        "opposite of",
        "opposite to",
        "out from",
        "out of",
        "outside of",
        #"owing to",
        #"prior to",
        #"pursuant to",
        #"rather than",
        #"regardless of",
        "right of",
        #"subsequent to",
        #"such as",
        #"thanks to",
        #"up to",
        #"as far as",
        #"as opposed to",
        #"as soon as",
        #"as well as",
        #"at the behest of",
        #"by means of",
        #"by virtue of",
        #"for the sake of",
        #"in accordance with",
        #"in addition to",
        #"in case of",
        "in front of",
        "in lieu of",
        #"in place of",
        "in point of",
        #"in spite of",
        #"on account of",
        #"on behalf of",
        "on top of",
        #"with regard to",
        #"with respect to",
        "with a view to",
]