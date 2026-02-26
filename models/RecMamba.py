import torch
from mamba_ssm import Mamba
from RevIN.RevIN import RevIN
from layers.Embed import Patching
from layers.FlattenHead import FlattenHead


### mamba n1 + patch n2 + 重构(x3+x4+res1) ###

class Model(torch.nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs


        patch_sizes = getattr(configs, 'patch_sizes', '')
        strides = getattr(configs, 'strides', '')
        if isinstance(patch_sizes, str) and patch_sizes.strip() != '':
            self.patch_sizes = [int(item.strip()) for item in patch_sizes.split(',') if item.strip() != '']
        else:
            self.patch_sizes = [configs.patch_size]

        if isinstance(strides, str) and strides.strip() != '':
            self.strides = [int(item.strip()) for item in strides.split(',') if item.strip() != '']
        else:
            self.strides = [configs.stride]

        if len(self.patch_sizes) != len(self.strides):
            raise ValueError('patch_sizes and strides must have the same number of elements.')

        self.patch_nums = [int((configs.seq_len - patch_size) / stride + 2)
                           for patch_size, stride in zip(self.patch_sizes, self.strides)]
        self.scale_num = len(self.patch_sizes)
        self.patchings = torch.nn.ModuleList([
            Patching(patch_size, stride, stride) for patch_size, stride in zip(self.patch_sizes, self.strides)
        ])

        if self.configs.revin == 1:
            self.revin_layer = RevIN(self.configs.enc_in)

        self.lin1 = torch.nn.Linear(self.configs.seq_len, self.configs.n1)
        self.dropout1 = torch.nn.Dropout(self.configs.dropout)

        self.lin2_layers = torch.nn.ModuleList([
            torch.nn.Linear(patch_size, self.configs.n2) for patch_size in self.patch_sizes
        ])
        self.dropout2_layers = torch.nn.ModuleList([
            torch.nn.Dropout(self.configs.dropout) for _ in range(self.scale_num)
        ])

        self.mamba1 = Mamba(d_model=self.configs.n1, d_state=self.configs.d_state, d_conv=self.configs.dconv,
                            expand=self.configs.e_fact)
        self.mamba1_r = Mamba(d_model=1, d_state=self.configs.d_state, d_conv=self.configs.dconv,
                              expand=self.configs.e_fact)

        self.mamba2_layers = torch.nn.ModuleList([
            Mamba(d_model=self.configs.n2, d_state=self.configs.d_state, d_conv=self.configs.dconv,
                  expand=self.configs.e_fact)
            for _ in range(self.scale_num)
        ])
        self.mamba2_r_layers = torch.nn.ModuleList([
            Mamba(d_model=patch_num, d_state=self.configs.d_state, d_conv=self.configs.dconv,
                  expand=self.configs.e_fact)
            for patch_num in self.patch_nums
        ])

        self.heads = torch.nn.ModuleList([
            FlattenHead(configs.enc_in, self.configs.n2 * patch_num, self.configs.n1, head_dropout=configs.head_dropout)
            for patch_num in self.patch_nums
        ])
        self.scale_logits = torch.nn.Parameter(torch.zeros(self.scale_num))
        self.lin3 = torch.nn.Linear(2 * self.configs.n1, self.configs.pred_len)
        self.lin4 = torch.nn.Linear(self.configs.n1, self.configs.seq_len)

        self.weight_limit = self.configs.w_limit

    def forward(self, x):
        x_origin = x
        if self.configs.revin == 1:
            x = self.revin_layer(x, 'norm')
        else:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        x = torch.permute(x, (0, 2, 1)) # B L D -> B D L

        x_for_patch = x

        x = torch.reshape(x, (x.shape[0] * x.shape[1], 1, x.shape[2])) # B D L -> (B * D) 1 L
        x_rep1 = self.lin1(x) # (B * D) 1 L -> (B * D) 1 n1
        x_res1 = x_rep1 # (B * D) 1 n1
        x_rep1 = self.dropout1(x_rep1)
        x1 = self.mamba1(x_rep1) # (B * D) 1 n1
        x1_r = self.mamba1_r(x_rep1.permute(0, 2, 1)).permute(0, 2, 1) # (B * D) 1 n1


        multi_scale_reps = []
        n_vars = None
        for idx in range(self.scale_num):
            x_patch, n_vars = self.patchings[idx](x_for_patch)  # B D L -> (B * D) N P
            x_rep2 = self.lin2_layers[idx](x_patch)
            x_res2 = x_rep2
            x_rep2 = self.dropout2_layers[idx](x_rep2)
            x2 = self.mamba2_layers[idx](x_rep2)
            x2_r = self.mamba2_r_layers[idx](x_rep2.permute(0, 2, 1)).permute(0, 2, 1)
            multi_scale_reps.append(self.heads[idx](x2 + x2_r + x_res2))

        scale_weights = torch.nn.functional.softmax(self.scale_logits, dim=0)
        x = torch.zeros_like(multi_scale_reps[0])
        for idx, rep in enumerate(multi_scale_reps):
            x = x + scale_weights[idx] * rep


        x = x.reshape(x.shape[0], 1, x.shape[1])  # (B * D) 1 n1
        x = torch.cat([x1 + x1_r, x + x_res1], dim=2)  # (B * D) 1 (2 * n1)

        x_pre = self.lin3(x).reshape(-1, n_vars, self.configs.pred_len).permute(0, 2, 1)
        x_rec = self.lin4(x1 + x1_r + x_res1).reshape(-1, n_vars, self.configs.seq_len).permute(0, 2, 1)

        if self.configs.revin == 1:
            x_pre = self.revin_layer(x_pre, 'denorm')
            x_rec = self.revin_layer(x_rec, 'denorm')
        else:
            x_pre = x_pre * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
            x_pre = x_pre + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.pred_len, 1))
            x_rec = x_rec * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.configs.seq_len, 1))
            x_rec = x_rec + (means[:, 0, :].unsqueeze(1).repeat(1, self.configs.seq_len, 1))

        ## get weight for reconstruction task
        ## 递减权重
        # diff = torch.abs(x_rec - x_origin)
        # diff_softmax = torch.nn.functional.softmax(diff, dim=-1)
        # max_diff = torch.max(diff_softmax, dim=-1).values
        # max_diff = torch.max(max_diff, dim=-1).values
        # current_weight = torch.mean(max_diff)
        # weight = torch.min(current_weight, torch.tensor(self.weight_limit))

        ## 增加权重的百分比
        # mean_origin = x_origin.mean()
        # std_origin = x_origin.std()
        # x_rec_normalized = (x_rec - mean_origin) / std_origin
        # x_origin_normalized = (x_origin - mean_origin) / std_origin
        # diff_percent = (torch.abs(x_rec_normalized - x_origin_normalized).mean() / torch.abs(x_origin_normalized).mean())
        # # filtered_diff = diff_percent[diff_percent <= 1]
        # weight = self.weight_limit * (1 + diff_percent)

        ## 权重本身的百分比
        diff = torch.abs(x_rec - x_origin)
        diff_softmax = torch.nn.functional.softmax(diff, dim=-1)
        max_diff = torch.max(diff_softmax, dim=-1).values
        max_diff = torch.max(max_diff, dim=-1).values
        weigth_precent = torch.mean(max_diff)
        weight = self.weight_limit * weigth_precent

        return x_pre, x_origin, x_rec, weight
