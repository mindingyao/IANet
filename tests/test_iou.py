"""Checks for IOU._iou, in particular all-background pred/target pairs.

Run with:  python3 tests/test_iou.py
Works with or without torch: IOU.py only uses torch.sum / nn.Module /
autograd.Variable, so when torch is missing a minimal numpy-backed stand-in
is installed before importing the untouched IOU.py source.
"""

import os
import sys
import types
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = types.ModuleType('torch')
    torch.sum = np.sum
    nn = types.ModuleType('torch.nn')

    class _StubModule(object):
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    nn.Module = _StubModule
    nn.functional = types.ModuleType('torch.nn.functional')
    autograd = types.ModuleType('torch.autograd')
    autograd.Variable = lambda x: x
    torch.nn = nn
    torch.autograd = autograd
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = nn
    sys.modules['torch.nn.functional'] = nn.functional
    sys.modules['torch.autograd'] = autograd

import IOU


def as_input(array):
    array = np.asarray(array, dtype=np.float32)
    return torch.from_numpy(array) if HAS_TORCH else array


def iou_loss(pred, target):
    return float(IOU._iou(as_input(pred), as_input(target)))


class TestIOULoss(unittest.TestCase):

    def test_all_background_is_finite_zero(self):
        pred = np.zeros((2, 1, 4, 4))
        target = np.zeros((2, 1, 4, 4))
        loss = iou_loss(pred, target)
        self.assertTrue(np.isfinite(loss), 'all-background loss is not finite: {}'.format(loss))
        self.assertAlmostEqual(loss, 0.0, places=6)

    def test_all_background_target_with_prediction_is_finite(self):
        pred = np.ones((1, 1, 2, 2))
        target = np.zeros((1, 1, 2, 2))
        loss = iou_loss(pred, target)
        self.assertTrue(np.isfinite(loss), 'empty target loss is not finite: {}'.format(loss))
        self.assertAlmostEqual(loss, 1.0 - 1.0 / 5.0, places=6)

    def test_partial_overlap_value(self):
        pred = np.array([[[[1.0, 0.0], [1.0, 0.0]]]])
        target = np.array([[[[1.0, 1.0], [0.0, 0.0]]]])
        # Iand = 1, Ior = 2 + 2 - 1 = 3, smoothed IoU = 2 / 4
        self.assertAlmostEqual(iou_loss(pred, target), 0.5, places=6)

    def test_perfect_overlap_value(self):
        pred = np.ones((3, 1, 5, 5))
        target = np.ones((3, 1, 5, 5))
        self.assertAlmostEqual(iou_loss(pred, target), 0.0, places=6)

    def test_mixed_batch_matches_mean(self):
        pred = np.zeros((2, 1, 2, 2))
        target = np.zeros((2, 1, 2, 2))
        pred[1] = np.array([[[1.0, 0.0], [1.0, 0.0]]])
        target[1] = np.array([[[1.0, 1.0], [0.0, 0.0]]])
        loss = iou_loss(pred, target)
        self.assertTrue(np.isfinite(loss), 'mixed batch loss is not finite: {}'.format(loss))
        self.assertAlmostEqual(loss, (0.0 + 0.5) / 2.0, places=6)

    def test_module_forward_all_background(self):
        criterion = IOU.IOU(size_average=True)
        pred = as_input(np.zeros((1, 1, 3, 3)))
        target = as_input(np.zeros((1, 1, 3, 3)))
        loss = float(criterion(pred, target))
        self.assertTrue(np.isfinite(loss), 'module forward loss is not finite: {}'.format(loss))
        self.assertAlmostEqual(loss, 0.0, places=6)

    @unittest.skipUnless(HAS_TORCH, 'torch is not installed')
    def test_gradients_are_finite_for_all_background(self):
        pred = as_input(np.zeros((1, 1, 4, 4))).requires_grad_(True)
        target = as_input(np.zeros((1, 1, 4, 4)))
        loss = IOU._iou(pred, target)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(pred.grad).all()), 'gradient contains NaN/Inf')


if __name__ == '__main__':
    unittest.main(verbosity=2)
