import torch

from lime_rec.models import SASRec
from scripts.training.train_sasrec import SeqDataset


def test_sasrec_predicts_full_catalog():
    model = SASRec(
        num_items=8,
        hidden=4,
        maxlen=3,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
    )
    model.eval()

    with torch.inference_mode():
        logits = model.predict(torch.tensor([[0, 1, 2]], dtype=torch.long))

    assert logits.shape == (1, 9)
    assert torch.isfinite(logits).all()


def test_paper_sampler_draws_one_unseen_negative_per_target():
    dataset = SeqDataset([[1, 2, 3, 4]], maxlen=5, num_items=7)
    _, targets, negatives = dataset[0]
    valid = targets != 0
    assert torch.all(negatives[~valid] == 0)
    assert torch.all((negatives[valid] >= 1) & (negatives[valid] <= 7))
    assert not ({1, 2, 3, 4} & set(negatives[valid].tolist()))


def test_paper_architecture_predicts_full_catalog():
    model = SASRec(8, 4, 3, 1, 1, 0.0, architecture="paper")
    model.eval()
    with torch.inference_mode():
        logits = model.predict(torch.tensor([[0, 1, 2]], dtype=torch.long))
    assert logits.shape == (1, 9)
    assert torch.isfinite(logits).all()
