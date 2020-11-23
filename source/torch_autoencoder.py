import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class Seq2Seq(nn.Module):
    def __init__(self, data_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=20, num_layers=1, batch_first=True)
        self.dense = nn.Linear(in_features=20, out_features=1)

    def forward(self, inp, h0, c0):
        output, (hn, cn) = self.lstm(input=inp, hx=(h0, c0))
        output = self.dense(output)
        return output, (hn, cn)


if __name__ == '__main__':
    loss_fn = nn.MSELoss()
    batchSize = 5
    timeSteps = 10
    epochs = 10
    seq_length = 50
    net = Seq2Seq(1, 1, 1)
    optimizer = optim.Adam(net.parameters(), lr=0.005)
    # (batch, seq_len, input_size)
    random_input = torch.randn(1000, seq_length, 1)
    plt.plot(range(seq_length), random_input[0].view(-1))
    # num_layers * num_directions, batch, hidden_size
    initial_h = torch.zeros(1, 1000, 20)
    initial_c = torch.zeros(1, 1000, 20)
    # net.zero_grad()
    for i in range(epochs):
        loss = 0
        hx = (initial_h, initial_c)
        whole_seq = None
        output, _ = net(random_input, *hx)
        loss = loss_fn(output, random_input)
        # curr_input = torch.zeros(1000, 1)
        # for k in range(seq_length):
        #     output, hx = net(curr_input.unsqueeze(2), *hx)
        #     if loss == 0:
        #         loss = loss_fn(output.squeeze(2), random_input[:, k, :])
        #     else:
        #         loss += loss_fn(output.squeeze(2), random_input[:, k, :])
        #     curr_input = random_input[:, k, :]

        print('{} {}'.format(i, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # evaluate
    start_width = 30
    curr_input = random_input[:, :start_width, :]
    total_seq = random_input[0, :start_width, :]

    with torch.no_grad():
        for k in range(seq_length - start_width):
            hx = (initial_h, initial_c)
            output, hx = net(curr_input, *hx)
            plt.plot(output[0].view(-1).numpy(), color='r')
            curr_input = torch.cat([curr_input, output[:, -1:, :]], dim=1)
            total_seq = torch.cat([total_seq, output[0, -1:, :]], dim=0)
            # output = torch.reshape(output, shape=[1000, 1, 1])
    # output, hx = net(curr_input, *(initial_h, initial_c))
    # plt.plot(range(seq_length), total_seq.view(-1))
    plt.show()

    print(output.shape, hx[0].shape, hx[1].shape)
