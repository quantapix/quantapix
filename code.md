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
