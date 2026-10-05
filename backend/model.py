"""CineVerse AI: hybrid neural recommender (collaborative + content towers)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
GENRES = ["Action", "Sci-Fi", "Romance", "Drama", "Comedy", "Thriller", "Animation"]

def synth(nu=600, ni=400, seed=0):
    """Sparse, popularity-skewed implicit feedback (swap in MovieLens for real data)."""
    r = np.random.default_rng(seed)
    G = (r.random((ni, 7)) < .2).astype(np.float32)
    G[G.sum(1) == 0, int(r.integers(0, 7))] = 1
    pop = np.minimum(r.zipf(1.6, ni), 200).astype(float); pop /= pop.sum()
    taste = r.dirichlet(np.ones(7) * .4, nu)
    rows = []
    for u in range(nu):
        p = pop * (.3 + G @ taste[u]); p /= p.sum()
        for i in r.choice(ni, 3 + int(r.exponential(10)), replace=False, p=p): rows.append((u, i))
    return np.array(rows), G, pop

class HybridNCF(nn.Module):
    def __init__(s, nu, ni, ng=7, d=32, p=.3):
        super().__init__()
        s.ue, s.ie = nn.Embedding(nu, d), nn.Embedding(ni, d)   # collaborative tower
        s.ut, s.it = nn.Linear(ng, d), nn.Linear(ng, d)         # content tower
        s.mlp = nn.Sequential(nn.Linear(4 * d, 64), nn.ReLU(), nn.Dropout(.2), nn.Linear(64, 1))
        s.p = p
    def forward(s, u, i, gi, gu, drop_u=False, drop_i=False):
        ue, ie = s.ue(u), s.ie(i)
        if s.training:  # ID dropout: forces the net to predict from content alone (cold-start)
            ue = ue * (torch.rand(len(u), 1) > s.p); ie = ie * (torch.rand(len(i), 1) > s.p)
        if drop_u: ue = torch.zeros_like(ue)
        if drop_i: ie = torch.zeros_like(ie)
        return s.mlp(torch.cat([ue * ie, s.ut(gu) * s.it(gi), ue, s.it(gi)], 1))

def train(epochs=8):
    pairs, G, pop = synth(); nu, ni = int(pairs[:, 0].max()) + 1, len(G)
    prof = np.zeros((nu, 7), np.float32)
    for u, i in pairs: prof[u] += G[i]
    prof /= prof.sum(1, keepdims=True) + 1e-9
    Gt, Pt, P = torch.tensor(G), torch.tensor(prof), torch.tensor(pairs)
    w_item = torch.tensor((pop.mean() / pop) ** .5, dtype=torch.float32)  # inverse-propensity weights
    m = HybridNCF(nu, ni); opt = torch.optim.Adam(m.parameters(), 2e-3)
    for _ in range(epochs):
        m.train()
        for b in torch.randperm(len(P)).split(512):
            u, i = P[b, 0], P[b, 1]; neg = torch.randint(0, ni, (len(b) * 3,))
            U, I = torch.cat([u, u.repeat(3)]), torch.cat([i, neg])
            y = torch.cat([torch.ones(len(u)), torch.zeros(len(neg))])
            w = torch.where(y == 1, w_item[I], torch.ones_like(y))   # de-bias popular positives
            loss = (F.binary_cross_entropy_with_logits(m(U, I, Gt[I], Pt[U]).squeeze(1), y, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m, G, pop, Pt

def _cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

@torch.no_grad()
def recommend(m, G, pop, Pt, k=8, user=None, genres=(), debias=.5):
    m.eval(); ni = len(G); I, Gt = torch.arange(ni), torch.tensor(G)
    if user is not None:
        u, gu, cold = torch.full((ni,), user), Pt[user].expand(ni, -1), False
    else:  # brand-new user: build a profile from chosen genres, no ID embedding
        v = np.zeros(7, np.float32)
        for g in genres: v[GENRES.index(g)] = 1
        u, gu, cold = torch.zeros(ni, dtype=torch.long), torch.tensor(v / v.sum()).expand(ni, -1), True
    s = torch.sigmoid(m(u, I, Gt, gu, drop_u=cold)).squeeze(1).numpy()
    s = s - debias * np.log1p(pop / pop.mean()) * s.std()          # penalise head items
    picked = []
    while len(picked) < k:                                           # MMR diversity re-rank
        picked.append(max((j for j in range(ni) if j not in picked), key=lambda j:
            s[j] - .2 * debias * max([_cos(G[j], G[p]) for p in picked], default=0)))
    return [{"item": int(j), "score": float(s[j]), "genres": [GENRES[g] for g in np.where(G[j])[0]]} for j in picked]
