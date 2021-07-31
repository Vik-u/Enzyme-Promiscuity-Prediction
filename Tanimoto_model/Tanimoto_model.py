#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Sequence similarity function
from Bio import pairwise2
from Bio.Align import substitution_matrices
from itertools import starmap




# Substrate similarity function
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity, BulkTanimotoSimilarity
from rdkit import Chem
from rdkit.Chem import AllChem


import pandas as pd
import numpy as np
import multiprocessing as mp

import argparse


# # Tanimoto model write-up and steps
# 
# Main idea: When compared to a given substrate, if a similar substrate interacts with an enzyme, then the given substrate will also interact with that and similar enzymes. 
# 
# Given an enzyme-substrate pair:
# 
# - Get smiles notation of substrate 
# - Select a substrate similarity threshold (Initialize and later optimize)
# - Detect substrates in the database which have Tanimoto similarity score with the given substrate greater than the selected substrate similarity threshold
# - Find enzymes known to react with those detected substrates
# - Select an enzyme similarity threshold (Initialize and later optimize)
# - Is the similarity score of the given enzyme with any of the found enzymes higher than the selected enzyme similarity threshold? 
#     - Yes: Enzyme reacts with the given substrate.
#     - No: It does not react with the given substrate.

# In[3]:


# Tanimoto model

class TanimotoInteractionPrediction:
    """Predict if an enzyme-substrate pair interacts based on their tanimoto similarity with
    existing interacting enzyme-substrate pairs"""
    def __init__(self, train_dti, train_enz, train_sub, 
                 valid_dti, valid_enz, valid_sub,
                 enz_sim_thresh=1, subs_sim_thresh=0.33,
                 with_label=True, start=0, end=250):
        
        # read training data
        self.train_dti_df_ = self._read_csv(train_dti, columns=[0,1,2,3])
        self.train_enz_df = self._read_csv(train_enz, columns=[1,2])
        self.train_sub_df = self._read_csv(train_sub, columns=[1,2])
        
        # get rid of illegal sequences from training
        illegal_proteins = list(self.train_enz_df.loc[self.train_enz_df.Sequence.str.contains("B|J|X|Z|O|U", regex=True)].index)
        self.train_dti_df = self.train_dti_df_.loc[~self.train_dti_df_.Protein_ID.isin(illegal_proteins)]
        
        # read validation data
        if with_label:
            self.valid_dti_df_ = self._read_csv(valid_dti, columns=[0,1,2,3]).iloc[start:end, :]
        else:
            self.valid_dti_df_ = self._read_csv(valid_dti, columns=[0,1,2]).iloc[start:end, :]
            
        self.valid_enz_df = self._read_csv(valid_enz, columns=[1,2])
        self.valid_sub_df = self._read_csv(valid_sub, columns=[1,2])
        
        # get rid of illegal sequences from validation
        illegal_proteins = list(self.valid_enz_df.loc[self.valid_enz_df.Sequence.str.contains("B|J|X|Z|O|U", regex=True)].index)
        self.valid_dti_df = self.valid_dti_df_.loc[~self.valid_dti_df_.Protein_ID.isin(illegal_proteins)]
        
        # modify training data to include only positive data
        self.pos_train_dti_df = self.train_dti_df.loc[self.train_dti_df.Label==1].drop_duplicates()
        
        # define the thresholds
        self.sst = subs_sim_thresh
        self.est = enz_sim_thresh
        
        # store substrate similarity values for memoization 
        self.sub_sim_dict = dict()
        self.enz_sim_dict = dict()
        pass
        
    
    
    def _read_csv(self, filename, columns):
        df = pd.read_csv(filename, index_col=0, usecols=columns)
        return df
        
    
    def _get_protein_similarity(self, seq1, seq2, matrix="BLOSUM62", gap_open=-10, gap_extend=-0.5):
        mat = substitution_matrices.load(name=matrix)
        alns = pairwise2.align.globalds(seq1, seq2, mat, gap_open, gap_extend)
        top_aln = alns[0]
        aln_human, aln_mouse, score, begin, end = top_aln
        return score/len(seq1)


    def _get_protein_bulk_similarity(self, seq1, list_of_seqs, matrix="BLOSUM62", gap_open=-10, gap_extend=-0.5):
        iterable = [(seq1, seq, matrix, gap_open, gap_extend) for seq in list_of_seqs]
        scores = starmap(self._get_protein_similarity, iterable)
        return np.array(list(scores))
    
    
    # get smiles to mf function
    def _smiles2fp(self, smiles):
        # get the molecule from smiles
        mol = Chem.MolFromSmiles(smiles)
        # get the molecular fingerprint as a 2048 length binary vector from molecule
        mf = Chem.RDKFingerprint(mol)
        return mf


    # write tanimoto calculator function for two smiles
    def _get_tanimoto_similarity(self, smiles1, smiles2):
        mf1 = self._smiles2fp(smiles1)
        mf2 = self._smiles2fp(smiles2)
        Tanimoto_score = TanimotoSimilarity(mf1, mf2)
        return Tanimoto_score


    # write tanimoto calculator function for multiple mfs
    def _get_bulk_tanimoto_similarity(self, smiles1, list_of_smiles):
        mf1 = self._smiles2fp(smiles1)
        mfs = list(map(self._smiles2fp, list_of_smiles))
        Tanimoto_scores = BulkTanimotoSimilarity(mf1, mfs)
        return np.array(Tanimoto_scores)
    
    
    # predict single enzyme substrate pair
    def predict_interaction(self, enz_id, sub_id):
        
        # get the substrate smiles and enzyme sequences from the validation df
        sub1_smiles = self.valid_sub_df.loc[sub_id, "smiles"]
        enz_seq = self.valid_enz_df.loc[enz_id, "Sequence"]
        
        # get smiles of existing substrates in the training df
        subs_smiles = self.train_sub_df.values.flatten()
        
        
        # memoization
#         if sub1_smiles in self.sub_sim_dict:
#             sim_subs = self.sub_sim_dict[sub1_smiles]
#         else:
#             # calculate bulk tanimoto similarity score
#             tss = self._get_bulk_tanimoto_similarity(sub1_smiles, subs_smiles)
#             # get substrates which have similarity score higher than threshold with the given substrate
#             sim_subs = list(self.train_sub_df.iloc[np.where(tss>self.sst)[0], :].index)
#             self.sub_sim_dict[sub1_smiles] = sim_subs
        
        
        # calculate bulk tanimoto similarity score
        tss = self._get_bulk_tanimoto_similarity(sub1_smiles, subs_smiles)
        # get substrates which have similarity score higher than threshold with the given substrate
        sim_subs = list(self.train_sub_df.iloc[np.where(tss>self.sst)[0], :].index)
        
        # get the enzymes which react with those substrates
        react_enzs = self.pos_train_dti_df.loc[self.pos_train_dti_df.Compound_ID.isin(sim_subs)].Protein_ID.values
        
        # if one of the reacting enzymes have similarity value higher than protein similarity threshold flag 
        # as reacting and break the loop
        for renz in react_enzs:
            renz_seq = self.train_enz_df.loc[renz, "Sequence"]
            
            # memoization
#             if enz_id in self.enz_sim_dict:
#                 if renz in self.enz_sim_dict[enz_id]:
#                     prot_sim_score = self.enz_sim_dict[enz_id][renz]
#                 else:
#                     prot_sim_score = self._get_protein_similarity(enz_seq, renz_seq)
#                     self.enz_sim_dict[enz_id][renz] = prot_sim_score
#                     if renz in self.enz_sim_dict:
#                         self.enz_sim_dict[renz][enz_id] = prot_sim_score
#                     else:
#                         self.enz_sim_dict[renz] = dict()
#                         self.enz_sim_dict[renz][enz_id] = prot_sim_score
#             else:
#                 prot_sim_score = self._get_protein_similarity(enz_seq, renz_seq)
#                 self.enz_sim_dict[enz_id] = dict()
#                 self.enz_sim_dict[enz_id][renz] = prot_sim_score
#                 if renz in self.enz_sim_dict:
#                     self.enz_sim_dict[renz][enz_id] = prot_sim_score
#                 else:
#                     self.enz_sim_dict[renz] = dict()
#                     self.enz_sim_dict[renz][enz_id] = prot_sim_score

            prot_sim_score = self._get_protein_similarity(enz_seq, renz_seq)
            
            if prot_sim_score>self.est:
                return 1
            
        return 0
    
    
    def bulk_prediction(self):
        pool = mp.Pool(mp.cpu_count())
        iterable = list(zip(self.valid_dti_df.Protein_ID.values, self.valid_dti_df.Compound_ID.values))
        results = pool.starmap(self.predict_interaction, iterable)
        return results
    
    
    def bulk_prediction_loop(self):
        result_list = []
        for enz, sub in zip(self.valid_dti_df.Protein_ID.values, self.valid_dti_df.Compound_ID.values):
            pred = self.predict_interaction(enz, sub)
            result_list.append(pred)
        return result_list


# In[4]:


# define training filenames
tr_dti = "../../../DeepConv-DTI/epp_examples/training_dataset/training_dti.csv"
tr_enz = "../../../DeepConv-DTI/epp_examples/training_dataset/training_protein.csv"
tr_sub = "../../../DeepConv-DTI/epp_examples/training_dataset/training_compound.csv"

# define validation filenames
va_dti = "../../../DeepConv-DTI/epp_examples/validation_dataset/validation_dti.csv"
va_enz = "../../../DeepConv-DTI/epp_examples/validation_dataset/validation_protein.csv"
va_sub = "../../../DeepConv-DTI/epp_examples/validation_dataset/validation_compound.csv"


# In[ ]:


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    # training file arguments
    parser.add_argument("tr_dti", help="training file with enz-subs interactions")
    parser.add_argument("tr_enz", help="training file with enz-sequence mappings")
    parser.add_argument("tr_sub", help="training file with subs-fingerprint mappings")

    # validation file arguments
    parser.add_argument("va_dti", help="validation file with enz-subs interactions")
    parser.add_argument("va_enz", help="validation file with enz-sequence mappings")
    parser.add_argument("va_sub", help="validation file with subs-fingerprint mappings")
    
    # optional arguments
    parser.add_argument("-t", "--est", help="enzyme similarity threshold", type=float, default=1.0)
    parser.add_argument("-T", "--sst", help="substrate similarity threshold", type=float, default=0.33)
    
    parser.add_argument("-l", "--labels", help="True if validation file contains labels", action="store_true")
    parser.add_argument("-s", "--start", help="validation file prediction start", type=int, default=0)
    parser.add_argument("-e", "--end", help="validation file prediction end", type=int, default=250)
    
    args = parser.parse_args()
    
    # define class
    tsm = TanimotoInteractionPrediction(args.tr_dti, args.tr_enz, args.tr_sub,
                                        args.va_dti, args.va_enz, args.va_sub,
                                        args.est, args.sst,
                                        args.labels, args.start, args.end)
    
    # bulk predictions
    res = tsm.bulk_prediction()
    
    
    # store file based on label argument and start and end 
    if args.labels:
        output_file = f"./valid-results/valid_{args.start}_{args.end}.csv"
    else:
        output_file = f"./kegg-results/kegg_{args.start}_{args.end}.csv"
        
    
    with open(output_file, "w") as f:
        f.write(",".join(list(map(str, res))))
        f.write("\n")
        
    
    if args.labels:
        with open(output_file, "a") as f:
            f.write(",".join(list(map(str, tsm.valid_dti_df.Label.values))))
            f.write("\n")


# In[ ]:




