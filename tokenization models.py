import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeEmbedding(nn.Module): # (t -> embed_dim)
    def __init__(self, embed_dim):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        dim = embed_dim // 2
        weights = torch.tensor([1 / (10000**(2*i/embed_dim)) for i in range(dim)])
        self.register_buffer('weights', weights)
        
    def forward(self, timestamps): # (B, N) -> (B, N, embed_dim)
        timestamps = timestamps.unsqueeze(-1) # (B, N, 1)
        weights = self.weights.view(1, 1, -1) # (1, 1, embed_dim/2)
        mat = weights * timestamps # (B, N, embed_dim/2)
        sins = torch.sin(mat)
        coss = torch.cos(mat)
        return torch.stack((sins, coss), dim=-1).flatten(-2) # (B, N, embed_dim)
    
class NoteEmbeddingContinuous(nn.Module): # (33 Vel, 88 ON, 88 OFF, t) -> note_embedding + time_embedding
    def __init__(self, embed_dim):
        super().__init__()
        self.time_embedder = TimeEmbedding(embed_dim)
        self.note_embedder = nn.Embedding(209, embed_dim)
        
    def forward(self, notes):
        times = notes[:, :, 1]
        times = self.time_embedder(times)
        notes = notes[:, :, 0]
        notes = self.note_embedder(notes)
        return times + notes
    
class SegmentEmbeddingContinuous(nn.Module): # (22 Val, 11 Amt, t) -> seg_embedding + time_embedding
    def __init__(self, embed_dim):
        super().__init__()
        self.time_embedder = TimeEmbedding(embed_dim)
        self.segment_embedder = nn.Embedding(33, embed_dim)
        
    def forward(self, segments):
        times = segments[:, :, 1]
        segments = segments[:, :, 0]
        times = self.time_embedder(times)
        segments = self.segment_embedder(segments)
        return times + segments
        
class TransformerCC(nn.Module): # [({209}, t), ({33}, t)] -> ({33}, t)
    def __init__(self, n_enc, n_dec, embed_dim, nhead):
        assert embed_dim % nhead == 0, "d_model must be diviseable by nhead"
        super().__init__()
        self.note_embedding = NoteEmbeddingContinuous(embed_dim)
        self.seg_embedding = SegmentEmbeddingContinuous(embed_dim)

        self.transformer = nn.Transformer(d_model=embed_dim, nhead=nhead, num_encoder_layers=n_enc,
                                          num_decoder_layers=n_dec, dim_feedforward=4*embed_dim, batch_first=True, norm_first=True)

        self.ffd = nn.Linear(embed_dim, 33)
        self.ffd2 = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1))
        
    def forward(self, notes, segs, note_mask=None, seg_mask=None):
        notes = self.note_embedding(notes)
        segs = self.seg_embedding(segs)
        L = segs.size(1)
        trig = nn.Transformer.generate_square_subsequent_mask(L, device=segs.device)
        segs = self.transformer(src=notes, tgt=segs, tgt_mask=trig, src_key_padding_mask=note_mask, tgt_key_padding_mask=seg_mask)
        return (self.ffd(segs), self.ffd2(segs))
    
class TransformerCT(nn.Module): # [({209}, t), ({33}, t)] -> {4033}
    def __init__(self, n_enc, n_dec, embed_dim, nhead):
        assert embed_dim % nhead == 0, "d_model must be diviseable by nhead"
        super().__init__()
        self.note_embedding = NoteEmbeddingContinuous(embed_dim)
        self.seg_embedding = SegmentEmbeddingContinuous(embed_dim)

        self.transformer = nn.Transformer(d_model=embed_dim, nhead=nhead, num_encoder_layers=n_enc,
                                          num_decoder_layers=n_dec, dim_feedforward=4*embed_dim, batch_first=True, norm_first=True)

        self.ffd = nn.Linear(embed_dim, 4033)
        
    def forward(self, notes, segs, note_mask=None, seg_mask=None):
        notes = self.note_embedding(notes)
        segs = self.seg_embedding(segs)
        L = segs.size(1)
        trig = nn.Transformer.generate_square_subsequent_mask(L, device=segs.device)
        segs = self.transformer(src=notes, tgt=segs, tgt_mask=trig, src_key_padding_mask=note_mask, tgt_key_padding_mask=seg_mask)
        return self.ffd(segs)
    
class TransformerTC(nn.Module): # ({4209}, {4033}) -> ({33}, t)
    def __init__(self, n_enc, n_dec, embed_dim, nhead):
        assert embed_dim % nhead == 0, "d_model must be diviseable by nhead"
        super().__init__()
        self.note_embedding = nn.Embedding(4209, embed_dim)
        self.seg_embedding = nn.Embedding(4033, embed_dim)

        self.transformer = nn.Transformer(d_model=embed_dim, nhead=nhead, num_encoder_layers=n_enc,
                                          num_decoder_layers=n_dec, dim_feedforward=4*embed_dim, batch_first=True, norm_first=True)

        self.ffd = nn.Linear(embed_dim, 33)
        self.ffd2 = nn.Sequential(nn.Linear(embed_dim, embed_dim // 2), nn.ReLU(), nn.Linear(embed_dim // 2, 1))
        
    def forward(self, notes, segs, note_mask=None, seg_mask=None):
        notes = self.note_embedding(notes)
        segs = self.seg_embedding(segs)
        L = segs.size(1)
        trig = nn.Transformer.generate_square_subsequent_mask(L, device=segs.device)
        segs = self.transformer(src=notes, tgt=segs, tgt_mask=trig, src_key_padding_mask=note_mask, tgt_key_padding_mask=seg_mask)
        return (self.ffd(segs), self.ffd2(segs))
    
class TransformerTT(nn.Module): # ({4209}, {4033}) -> {4033}
    def __init__(self, n_enc, n_dec, embed_dim, nhead):
        assert embed_dim % nhead == 0, "d_model must be diviseable by nhead"
        super().__init__()

        self.note_embedding = nn.Embedding(4209, embed_dim)
        self.seg_embedding = nn.Embedding(4033, embed_dim)

        self.transformer = nn.Transformer(d_model=embed_dim, nhead=nhead, num_encoder_layers=n_enc,
                                          num_decoder_layers=n_dec, dim_feedforward=4*embed_dim, batch_first=True, norm_first=True)

        self.ffd = nn.Linear(embed_dim, 4033)
        
    def forward(self, notes, segs, note_mask=None, seg_mask=None):
        notes = self.note_embedding(notes)
        segs = self.seg_embedding(segs)
        L = segs.size(1)
        trig = nn.Transformer.generate_square_subsequent_mask(L, device=segs.device)
        segs = self.transformer(src=notes, tgt=segs, tgt_mask=trig, src_key_padding_mask=note_mask, tgt_key_padding_mask=seg_mask)
        return self.ffd(segs)
    
