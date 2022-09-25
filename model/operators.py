import cvxpy as cp
import geoopt
import torch
import mcr2.functional as F


class LinearUnconstrainedEncoder(torch.nn.Module):
    def __init__(self, d_x, d_z, lr):
        super(LinearUnconstrainedEncoder, self).__init__()
        self.matrix = torch.nn.Parameter(torch.randn(d_z, d_x))
        self.optimizer = torch.optim.Adam([self.matrix], lr=lr)

    def forward(self, X):
        return X @ self.matrix.T

    def project_parameters(self, X, Pi):
        pass


class LinearUnconstrainedDecoder(torch.nn.Module):
    def __init__(self, d_x, d_z, lr):
        super(LinearUnconstrainedDecoder, self).__init__()
        self.matrix = torch.nn.Parameter(torch.randn(d_x, d_z))
        self.optimizer = torch.optim.Adam([self.matrix], lr=lr)

    def forward(self, Z):
        return Z @ self.matrix.T

    def project_parameters(self, X, Pi):
        pass


class LinearOrthogonallyConstrainedEncoder(torch.nn.Module):
    def __init__(self, d_x, d_z, lr):
        super(LinearOrthogonallyConstrainedEncoder, self).__init__()
        self.parameter_manifold = geoopt.Stiefel()
        if d_z >= d_x:
            self.matrix = geoopt.ManifoldParameter(self.parameter_manifold.random(d_z, d_x))
        elif d_z < d_x:
            self.matrix = geoopt.ManifoldParameter(self.parameter_manifold.random(d_x, d_z).T)
        self.optimizer = geoopt.optim.RiemannianSGD([self.matrix], lr=lr)

    def forward(self, X):
        return X @ self.matrix.T

    def project_parameters(self, X, Pi):
        self.matrix.data = self.parameter_manifold.projx(self.matrix.data)


class LinearOrthogonallyConstrainedDecoder(torch.nn.Module):
    def __init__(self, d_x, d_z, lr):
        super(LinearOrthogonallyConstrainedDecoder, self).__init__()
        self.parameter_manifold = geoopt.Stiefel()
        if d_z >= d_x:
            self.matrix = geoopt.ManifoldParameter(self.parameter_manifold.random(d_z, d_x).T)
        elif d_z < d_x:
            self.matrix = geoopt.ManifoldParameter(self.parameter_manifold.random(d_x, d_z))
        self.optimizer = geoopt.optim.RiemannianAdam([self.matrix], lr=lr)

    def forward(self, Z):
        return Z @ self.matrix.T

    def project_parameters(self, X, Pi):
        self.matrix.data = self.parameter_manifold.projx(self.matrix.data)


class LinearSampleNormConstrainedEncoder(torch.nn.Module):
    def __init__(self, d_x, d_z, lr):
        super(LinearSampleNormConstrainedEncoder, self).__init__()
        self.matrix = torch.nn.Parameter(torch.randn(d_z, d_x))
        self.optimizer = torch.optim.Adam([self.matrix], lr=lr)

    def forward(self, X):
        return X @ self.matrix.T

    def project_parameters(self, X, Pi):
        F = self.matrix.data
        pi_F = cp.Variable(shape=F.shape)
        constraints = []
        for i in range(Pi.shape[1]):
            Xi = X[Pi[:, i] == 1].detach().numpy()
            if Xi.shape[0] > 0:
                constraints.append(cp.norm(pi_F @ Xi.T, 'fro') ** 2 <= Xi.shape[0])
        objective = cp.Minimize(cp.norm(pi_F - F, 'fro'))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)
        self.matrix.data.fill_(0.0)
        self.matrix.data += torch.tensor(pi_F.value, dtype=F.dtype)


class FCNNEncoder(torch.nn.Module):
    def __init__(self, d_x, d_z, d_latent, n_layers, lr=1e-3):
        super(FCNNEncoder, self).__init__()
        layers = [torch.nn.Linear(d_x, d_latent), torch.nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([torch.nn.Linear(d_latent, d_latent), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(d_latent, d_z))
        self.net = torch.nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, X):
        return self.net(X)


class FCNNDecoder(torch.nn.Module):
    def __init__(self, d_x, d_z, d_latent, n_layers, lr=1e-3):
        super(FCNNDecoder, self).__init__()
        layers = [torch.nn.Linear(d_z, d_latent), torch.nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([torch.nn.Linear(d_latent, d_latent), torch.nn.ReLU()])
        layers.append(torch.nn.Linear(d_latent, d_x))
        self.net = torch.nn.Sequential(*layers)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, Z):
        return self.net(Z)
