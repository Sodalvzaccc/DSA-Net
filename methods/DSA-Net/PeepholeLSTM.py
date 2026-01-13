import torch
import torch.nn as nn
import torch.nn.functional as F

class PeepholeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PeepholeLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))

        hh_weights = torch.zeros(4 * hidden_size, hidden_size)
        hh_weights[:3 * hidden_size] = torch.randn(3 * hidden_size, hidden_size)
        self.weight_hh = nn.Parameter(hh_weights)

        ch_weights = torch.zeros(4 * hidden_size, hidden_size)
        ch_weights[:3 * hidden_size] = torch.randn(3 * hidden_size, hidden_size)
        self.weight_ch = nn.Parameter(ch_weights)

        self.bias = nn.Parameter(torch.randn(4 * hidden_size))
        

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / self.hidden_size ** 0.5
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input, state):
        hx, cx = state
        
        gates = F.linear(input, self.weight_ih, self.bias) + \
                F.linear(hx, self.weight_hh) + \
                F.linear(cx, self.weight_ch)
                
        ingate, forgetgate, outgate, cellgate = gates.chunk(4, 1)
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        outgate = torch.sigmoid(outgate)
        cellgate = torch.tanh(cellgate)
        
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)
        
        return hy, cy



class PeepholeLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(PeepholeLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first


        self.cells = nn.ModuleList([PeepholeLSTMCell(input_size, hidden_size) for _ in range(1)]) 

    def forward(self, inputs, initial_states=None):

        if self.batch_first:
            inputs = inputs.transpose(0, 1)  
        sequence_length, batch_size, _ = inputs.size()

        if initial_states is None:
            hx = inputs.new_zeros(batch_size, self.hidden_size)
            cx = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            hx, cx = initial_states

        outputs = []
        for t in range(sequence_length):
            hx, cx = self.cells[0](inputs[t], (hx, cx))
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)
        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        return outputs



class BiPeepholeLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BiPeepholeLSTMCell, self).__init__()

        self.forward_cell = PeepholeLSTMCell(input_size, hidden_size)

        self.backward_cell = PeepholeLSTMCell(input_size, hidden_size)

    def forward(self, inputs, forward_state, backward_state):
        forward_hx, forward_cx = forward_state
        backward_hx, backward_cx = backward_state

        forward_hy, forward_cy = self.forward_cell(inputs, (forward_hx, forward_cx))

        backward_hy, backward_cy = self.backward_cell(inputs, (backward_hx, backward_cx))

        return (forward_hy, forward_cy), (backward_hy, backward_cy)


class BiPeepholeLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(BiPeepholeLSTMLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.bi_cell = BiPeepholeLSTMCell(input_size, hidden_size)

    def forward(self, inputs, initial_states=None):
        if self.batch_first:
            inputs = inputs.transpose(0, 1) 
        sequence_length, batch_size, _ = inputs.size()

        if initial_states is None:
            forward_hx = inputs.new_zeros(batch_size, self.hidden_size)
            forward_cx = inputs.new_zeros(batch_size, self.hidden_size)
            backward_hx = inputs.new_zeros(batch_size, self.hidden_size)
            backward_cx = inputs.new_zeros(batch_size, self.hidden_size)
        else:
            forward_hx, forward_cx, backward_hx, backward_cx = initial_states

        forward_outputs = []
        backward_outputs = []
        for t in range(sequence_length):
            (forward_hx, forward_cx), (backward_hx, backward_cx) = self.bi_cell(
                inputs[t], (forward_hx, forward_cx), (backward_hx, backward_cx))
            forward_outputs.append(forward_hx.unsqueeze(0))
            backward_outputs.append(backward_hx.unsqueeze(0))


        forward_outputs = torch.cat(forward_outputs, dim=0)
        backward_outputs = torch.cat(backward_outputs[::-1], dim=0) 

        if self.batch_first:
            forward_outputs = forward_outputs.transpose(0, 1)
            backward_outputs = backward_outputs.transpose(0, 1)

        outputs = torch.cat([forward_outputs, backward_outputs], dim=-1)
        return outputs


if __name__ == "__main__":
    cell = PeepholeLSTMCell(512, 1024)
    input = torch.randn(1, 512)  
    hx = torch.randn(1, 1024)     
    cx = torch.randn(1, 1024)    

    hy, cy = cell(input, (hx, cx))
    print(hy.shape, cy.shape) 