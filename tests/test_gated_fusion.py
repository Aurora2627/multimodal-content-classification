import unittest
import torch
from gated_fusion import GatedMLPFusion
from torch_models import build_head
from accelerated_training import train_head, load_head

class GatedFusionTests(unittest.TestCase):
    def test_matched_initialization_rng_and_gate_gradient(self):
        torch.manual_seed(42)
        baseline=build_head(12, 8).eval()
        expected_rng=torch.random.get_rng_state().clone()
        torch.manual_seed(42)
        gated=GatedMLPFusion(12, 8).eval()
        self.assertTrue(torch.equal(torch.random.get_rng_state(), expected_rng))
        x=torch.randn(12,12)
        torch.testing.assert_close(gated(x), baseline(x), rtol=0, atol=0)
        gated(x).square().sum().backward()
        self.assertGreater(gated.gate[-1].weight.grad.abs().sum().item(), 0)
        torch.testing.assert_close(gated.gate_weights(x).sum(1), torch.ones(12))
        with self.assertRaises(ValueError):GatedMLPFusion(11)

    def test_training_reload_and_neutral_ablation(self):
        x=torch.cat([torch.eye(6), torch.eye(6)],1).repeat(3,1)
        y=torch.arange(6).repeat(3)
        for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
            model,info=train_head(x,y,x,y,device,hidden=8,epochs=3,architecture='gated-mlp')
            self.assertGreater(info['optimizer_steps'],0)
            bundle={'input_dim':12,'hidden_dim':8,'architecture':'gated-mlp',
                    'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()}}
            restored=load_head(bundle,device);dx=x.to(device)
            torch.testing.assert_close(model(dx),restored(dx),rtol=0,atol=0)
            torch.testing.assert_close(model(dx,neutral_gate=True),model.classifier(dx),rtol=0,atol=0)
