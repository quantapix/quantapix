# Relevant Python samples

## BART and mBART encoder layer differences

```python
def bart_enc_layer_forward(self, x, y_attn=False, **kw):
    yo = self.get_y_opts(y_attn=y_attn, **kw)
    y, a, _ = self.refl(x, yo=yo, **kw)
    y = self.drop(y)
    y = self.norm_refl(x + y)
    x = y
    y = self.drop_act(self.act(self.ff(y)))
    y = self.drop(self.proj(y))
    y = x + y
    y = self.norm(y)
    if y.dtype == torch.float16 and (torch.isinf(y).any() or torch.isnan(y).any()):
        clamp = torch.finfo(y.dtype).max - 1000
        y = torch.clamp(y, min=-clamp, max=clamp)
    y = (y,)
    if yo.attn:
        y += (a,)
    return y

```

```python
def mbart_enc_layer_forward(self, x, y_attn=False, **kw):
    yo = self.get_y_opts(y_attn=y_attn, **kw)
    y = self.norm_refl(x)
    y, a, _ = self.refl(y, yo=yo, **kw)
    y = x + self.drop(y)
    x = y
    y = self.norm(y)
    y = self.drop_act(self.act(self.ff(y)))
    y = self.drop(self.proj(y))
    y = x + y
    if y.dtype == torch.float16 and (torch.isinf(y).any() or torch.isnan(y).any()):
        clamp = torch.finfo(y.dtype).max - 1000
        y = torch.clamp(y, min=-clamp, max=clamp)
    y = (y,)
    if yo.attn:
        y += (a,)
    return y
```

## GPT2 attention forward function

```python
def gpt2_attention_forward(self, x, mask, head_m, enc=None, enc_m=None, prev_kv=None, **kw):
    cfg = self.cfg
    yo = self.get_y_opts(**kw)
    if enc is None:
        q, k, v = self.c_attn(x).split(cfg.d_hidden, dim=2)
    else:
        q = self.attn(x)
        k, v = self.c_attn(enc).split(cfg.d_hidden, dim=2)
        mask = enc_m
    q = self.split_heads(q)
    k = self.split_heads(k)
    v = self.split_heads(v)
    if prev_kv is not None:
        k = torch.cat((prev_kv[0], k), dim=-2)
        v = torch.cat((prev_kv[1], v), dim=-2)
    kv = (k, v) if yo.cache else None
    if cfg.reorder:
        ys = self.reordered(q, k, v, mask, head_m, yo=yo)
    else:
        ys = self.scores(q, k, v, mask, head_m, yo=yo)
    y = self.join_heads(ys[0])
    y = self.drop(self.proj(y))
    y = [y, kv] + ys[1:]
    return y
```
