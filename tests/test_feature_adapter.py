import unittest
import torch
from accelerated_training import train_head, load_head
from feature_adapter import FeatureAdapterFusion

class FeatureAdapterTests(unittest.TestCase):
    def test_identity_initialization_and_gradients(self):
        model = FeatureAdapterFusion(12, 4)
        x = torch.randn(12, 12)
        torch.testing.assert_close(model(x), model.classifier(x), rtol=0, atol=0)
        model(x).sum().backward()
        self.assertGreater(model.text_adapter[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.image_adapter[-1].weight.grad.abs().sum().item(), 0)

    def test_training_and_checkpoint(self):
        x = torch.cat([torch.eye(6), torch.eye(6)], dim=1).repeat(3, 1)
        y = torch.arange(6).repeat(3)
        for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
            model, info = train_head(x, y, x, y, device, hidden=4, epochs=3,
                                     architecture="feature-adapter")
            self.assertGreater(info["optimizer_steps"], 0)
            bundle = {"input_dim":12,"hidden_dim":4,"architecture":"feature-adapter",
                      "state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()}}
            restored = load_head(bundle, device)
            torch.testing.assert_close(model(x.to(device)), restored(x.to(device)), rtol=0, atol=0)
