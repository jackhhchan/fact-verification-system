import torch
from torch import nn
import numpy as np


class Head(nn.Module):
    def __init__(self, embedding_length: int, interaction: bool):
        super().__init__()
        self._interaction = interaction

        concatenation_scale: int
        if self._interaction:
            concatenation_scale = 3
        else:
            concatenation_scale = 2
        self.linear_0 = nn.Linear(in_features=embedding_length * concatenation_scale, out_features=128, bias=True)
        # self.act_0 = nn.Sigmoid()
        self.act_0 = nn.LeakyReLU()
        self.linear_1 = nn.Linear(in_features=self.linear_0.out_features, out_features=2, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, premises, hypothesis):
        assert premises.shape == hypothesis.shape, "Premise and hypothesis shape does not match up."
        if self._interaction:
            interaction = torch.sub(premises, hypothesis)
            concatenated = torch.cat((premises, hypothesis, interaction), dim=1)
        else:
            concatenated = torch.cat((premises, hypothesis), dim=1)
        final = self.linear_1(self.act_0(self.linear_0(concatenated)))
        return self.softmax(final)


if __name__ == '__main__':
    premise = torch.tensor(data=np.ones((1, 4)), dtype=torch.float32)
    hyp = torch.tensor(data=np.ones((1, 4)) + 1, dtype=torch.float32)
    print(premise, hyp)

    # optional - interaction feature: minus each other
    interaction = torch.sub(premise, hyp)
    print(interaction)

    concatenated = torch.cat((premise, hyp, interaction), dim=1)
    print(concatenated)

    linear = nn.Linear(in_features=concatenated.shape[1], out_features=2)
    final = linear(concatenated)
    print(final)

    softmax = nn.Softmax(dim=1)
    print(softmax(final))

    # optional: LogSoftmax?
    print("Using head...")
    head = Head(embedding_length=premise[0].shape[0], interaction=True)
    print(head(premise, hyp))
