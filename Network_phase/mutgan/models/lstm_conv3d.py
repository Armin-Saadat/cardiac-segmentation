import torch.nn as nn


class LSTM_Conv3D(nn.Module):
    def __init__(self, seq_len, height, width, stages_num):
        super(LSTM_Conv3D, self).__init__()
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.stages_num = stages_num

        assert (seq_len - 1) % stages_num == 0, 'stages_num choices: [1, 2, 3, 4, 6, 8, 12, 24]'
        t_kernel = ((seq_len - 1) // stages_num) + 1

        blocks = []
        for i in range(self.stages_num):
            blocks.append(nn.LSTM(input_size=width * height, hidden_size=width * height, batch_first=True))
            blocks.append(nn.Conv3d(in_channels=1, out_channels=1, kernel_size=(t_kernel, 3, 3), padding=(0, 1, 1)))
        self.blocks = nn.ModuleList(blocks)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # shape of inputs: (bs, seq_len, H, W)
        bs = inputs.shape[0]
        seq_len = self.seq_len

        lstm_out, (h, c) = self.blocks[0](inputs.view(bs, seq_len, -1))
        conv_out = self.blocks[1](lstm_out.view(bs, seq_len, self.height, self.width).unsqueeze(1))

        i = 2
        while i < self.stages_num * 2:
            seq_len = conv_out.shape[2]
            lstm_out, (h, c) = self.blocks[i](conv_out.squeeze(1).view(bs, seq_len, -1), (h, c))
            conv_out = self.blocks[i + 1](lstm_out.view(bs, seq_len, self.height, self.width).unsqueeze(1))
            i += 2

        # shape of output: (bs, H, W)
        out = conv_out.squeeze(1).squeeze(1)
        outputs = self.sigmoid(out).view(bs, self.height, self.width)

        return outputs
