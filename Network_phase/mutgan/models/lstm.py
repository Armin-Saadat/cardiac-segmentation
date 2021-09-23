import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, seq_len, height, width, stages_num):
        super(LSTM, self).__init__()
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.stages_num = stages_num

        blocks = []
        for i in range(self.stages_num):
            blocks.append(nn.LSTM(input_size=width * height, hidden_size=width * height, batch_first=True))
        self.blocks = nn.ModuleList(blocks)

        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        # shape of inputs: (bs, seq_len, H, W)
        bs = inputs.shape[0]

        lstm_out, (h, c) = self.blocks[0](inputs.view(bs, self.seq_len, -1))
        for i in range(1, self.stages_num):
            lstm_out, (h, c) = self.blocks[i](lstm_out, (h, c))

        # shape of output: (bs, H, W)
        out = lstm_out[:, -1, :]
        outputs = self.sigmoid(out).view(bs, self.height, self.width)

        return outputs
