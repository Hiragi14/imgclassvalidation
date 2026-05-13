# tests/test_count_params.py

import torch.nn as nn

from imgclassvalidation.extras import count_params


def test_count_params_all_trainable():
    model = nn.Sequential(
        nn.Linear(4, 3),  # weight: 4*3 = 12, bias: 3
        nn.ReLU(),
        nn.Linear(3, 2),  # weight: 3*2 = 6, bias: 2
    )

    result = count_params(model)

    # total = 12 + 3 + 6 + 2 = 23
    assert result == {
        "params_total": 23,
        "params_trainable": 23,
    }


def test_count_params_with_frozen_layer():
    model = nn.Sequential(
        nn.Linear(4, 3),  # 15 params
        nn.ReLU(),
        nn.Linear(3, 2),  # 8 params
    )

    # 1層目をfreeze
    for p in model[0].parameters():
        p.requires_grad = False

    result = count_params(model)

    assert result == {
        "params_total": 23,
        "params_trainable": 8,
    }


def test_count_params_no_parameters():
    model = nn.ReLU()

    result = count_params(model)

    assert result == {
        "params_total": 0,
        "params_trainable": 0,
    }


def test_count_params_all_frozen():
    model = nn.Linear(4, 3)  # weight: 12, bias: 3

    for p in model.parameters():
        p.requires_grad = False

    result = count_params(model)

    assert result == {
        "params_total": 15,
        "params_trainable": 0,
    }
