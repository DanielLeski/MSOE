from unittest import TestCase
import layers
import numpy as np
import torch
import unittest
from layers import *
from network import * 


class TestSum(TestCase):
    def setUp(self):
        self.a = layers.Input((2,1), train=True)
        self.a.set(torch.tensor([[3],[5]],dtype=torch.float64))
        self.b = layers.Input((2,1), train=True)
        self.b.set(torch.tensor([[1],[2]],dtype=torch.float64))
        self.sum = layers.Sum(self.a, self.b)

    def test_forward(self):
        self.sum.forward()
        assert torch.allclose(self.sum.output, torch.tensor([[4], [7]], dtype=torch.float64))
        
    def test_backward(self):
        self.sum.forward()
        self.sum.accumulate_grad(torch.ones(2,1))
        self.sum.backward()
        #print(self.a.grad)
        #print(self.b.grad)
        assert torch.allclose(self.a.grad, torch.ones((2,1)))
        assert torch.allclose(self.b.grad, torch.ones((2,1)))
        
    def test_step(self):
        self.sum.forward()
        self.sum.accumulate_grad(torch.ones((2,1)))
        self.sum.backward()
        self.a.step(step_size=0.1)
        self.b.step(step_size=0.1)
        #print(self.a.output)
        #print(self.b.output)
        assert torch.allclose(self.a.output, torch.tensor([[2.9], [4.9]], dtype=torch.float64))
        assert torch.allclose(self.b.output, torch.tensor([[0.9], [1.9]], dtype=torch.float64))


if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)


