class MolSig:
    """Protein Molecular Signature"""
    def __init__(self, p_dist=5):
        self.encoder_dict = dict()
        self.p_dist = p_dist
        
    
    def fit(self, x_train):
        all_sigs = list(map(self.get_mol_sig, x_train))
        sig_set = set()
        for sigs in all_sigs:
            for sig in sigs:
                sig_set.add(sig)
        self.encoder_dict = dict(zip(sig_set, range(len(sig_set))))
        return
    
    
    def get_encoding(self, sigs):
        cols = []
        for sig in sigs:
            if sig in self.encoder_dict:
                cols.append(self.encoder_dict[sig])
        return cols
    
    
    def transform(self, x_valid):
        if not self.encoder_dict:
            raise ValueError("Need to fit first")
        
        else:
            data = []
            row = []
            col = []
            for row_idx,seq in enumerate(x_valid):
                seq_sigs = self.get_mol_sig(seq, self.p_dist)
                seq_col = self.get_encoding(seq_sigs)
                for col_idx in seq_col:
                    data.append(1)
                    row.append(row_idx)
                    col.append(col_idx)
            return coo_matrix((data,(row, col)), shape=(len(x_valid), len(self.encoder_dict)))
        
    
    def get_mol_sig(self, prot_seq, p_dist=5):
        
        """Get the molecular signatures for a protein sequence"""
    
        psig = []

        len_seq = len(prot_seq)

        for i in range(len_seq):
            if i < p_dist:
                psig.append(prot_seq[0:i+p_dist + 1])
            elif len_seq - i <= p_dist:
                psig.append(prot_seq[i-p_dist:len_seq])
            else:
                psig.append(prot_seq[i-p_dist:i+p_dist + 1])

        return psig
