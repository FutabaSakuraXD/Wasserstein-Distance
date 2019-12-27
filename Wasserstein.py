class WassersteinDistance(nn.Module):
    r"""
    Given two empirical measures with n points each with locations x and y,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'

    Shape:
        - Input: :math:`(N, \text{in\_features})`, :math:`(N, \text{in\_features})`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps=0.01, max_iter=100, reduction='mean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        C = C.cuda()
        n_points = x.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, n_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / n_points).squeeze()
        nu = torch.empty(batch_size, n_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / n_points).squeeze()

        u = torch.zeros_like(mu)
        u = u.cuda()
        v = torch.zeros_like(nu)
        v = v.cuda()
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8).cuda() - self.lse(self.M(C, u, v))).cuda() + u
            v = self.eps * (torch.log(nu+1e-8).cuda() - self.lse(self.M(C, u, v).transpose(-2, -1))).cuda() + v
            err = (u - u1).abs().sum(-1).mean()
            err = err.cuda()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def lse(A):
        "log-sum-exp"
        # add 10^-6 to prevent NaN
        result = torch.log(torch.exp(A).sum(-1) + 1e-6)
        return result

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1
if __name__ == '__main__':
    x = torch.randn(16,395)
    y = torch.randn(16,395)
    softmax = torch.nn.Softmax(dim=1)
    W = WassersteinDistance()
    dist = W(softmax(x),softmax(y))
