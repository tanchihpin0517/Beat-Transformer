import torch
from beat_transformer import DemixedDilatedTransformerModel


@torch.no_grad()
def test_model():
    x = torch.randn(1, 5, 4096, 128)
    model = DemixedDilatedTransformerModel(
        attn_len=5, instr=5, ntoken=2,
        dmodel=256, nhead=8, d_hid=1024,
        nlayers=9, norm_first=True
    )
    model.eval()
    if torch.cuda.is_available():
        x = x.cuda()
        model.cuda()
    y = model(x)


if __name__ == "__main__":
    test_model()
