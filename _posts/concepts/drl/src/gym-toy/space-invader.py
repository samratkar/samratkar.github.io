env = gym.make("ALE/SpaceInvaders-V5")
class NN(nn.Module):
    def __init__(self, dim_inputs, dim_outputs):