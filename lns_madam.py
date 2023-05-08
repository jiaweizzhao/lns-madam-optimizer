import torch
from torch.optim.optimizer import Optimizer, required


class LNS_Madam(Optimizer):

    def __init__(self, params, lr=1/128, p_scale=3.0, g_bound=10.0, wd=None, momentum=0):

        self.p_scale = p_scale
        self.g_bound = g_bound
        self.wd = wd
        self.momentum = momentum
        self.dampening = 0
        defaults = dict(lr=lr)
        super(LNS_Madam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                d_p = p.grad.data

                if len(state) == 0:
                    state['max'] = self.p_scale*(p*p).mean().sqrt().item()
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1
                bias_correction = 1 - 0.999 ** state['step']
                state['exp_avg_sq'] = 0.999 * state['exp_avg_sq'] + 0.001 * p.grad.data**2
                
                g_normed = d_p / (state['exp_avg_sq']/bias_correction).sqrt()
                g_normed[torch.isnan(g_normed)] = 0
                g_normed.clamp_(-self.g_bound, self.g_bound)
                g_normed.round_() #rounded 
                
                if self.wd is not None: 
                    p.data *= torch.exp( -group['lr']*g_normed*torch.sign(p.data) - group['lr']*self.wd )
                else:
                    #p.data *= torch.exp( -group['lr']*g_normed*torch.sign(p.data) )
                    p.data *= 2.0**( -group['lr']*g_normed*torch.sign(p.data))
                p.data.clamp_(-state['max'], state['max'])

        return loss