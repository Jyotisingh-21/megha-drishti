import numpy as np
import torch
import xarray as xr

from nwpblend.blend.gating import GatingBlender, GatingNetwork


def test_gating_network_masking_and_sum():
    n_models = 4
    in_features = 10
    model = GatingNetwork(in_features, n_models)

    x = torch.randn(2, in_features)
    # Batch 0: all available
    # Batch 1: model 1 and 3 available, 0 and 2 missing
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])

    weights = model(x, mask)

    # 1. Non-negative
    assert torch.all(weights >= 0)

    # 2. Sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))

    # 3. Missing models get 0
    assert weights[1, 0] == 0.0
    assert weights[1, 2] == 0.0


def test_gating_overfit():
    # Construct a tiny dataset where model 2 is always perfectly correct
    n_models = 3
    in_features = 2

    # We feed forecast values directly as features to test if it learns
    X = torch.rand(100, in_features)
    F = torch.rand(100, n_models)
    M = torch.ones(100, n_models)

    # Make model 2 perfect
    Y = F[:, 2].clone()

    blender = GatingBlender(n_models=n_models, is_precip=False)
    # Fit
    blender.fit(X, Y, M, F, epochs=50, lr=0.1)

    # Predict
    weights = blender.predict_weights(X, M)

    # It should have learned to heavily weight model 2
    assert weights[:, 2].mean() > 0.8, (
        f"Failed to overfit on the perfect model, weight is {weights[:, 2].mean()}"
    )
