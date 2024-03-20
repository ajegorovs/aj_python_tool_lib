
import torch.nn as nn

class LSTM_torch(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LSTM_torch, self).__init__()
        self.hidden_size = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.lin = nn.Linear(hidden_dim, input_dim)

    def forward(self, inp, h0 = None, c0 = None):
        
        if h0 is None or c0 is None:
            lstm_out, (hs, cs) = self.lstm(inp)
        else:
            lstm_out, (hs, cs)= self.lstm(inp, (h0, c0))

        output = self.lin(lstm_out[[-1]])
        
        return output, hs, cs 
    

class FakeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, fn1, fn2):
        super(FakeLSTM, self).__init__()

        self.hidden_size      = hidden_size
        self.init_weights     = fn1
        self.init_states      = fn2
        self.shape_Wi_stack   = (4*hidden_size, input_size )
        self.shape_Wh_stack   = (4*hidden_size, hidden_size)
        self.shape_b_stack    = (4*hidden_size, 1          )

        self.Wi_stack   = nn.Parameter(self.init_weights(self.shape_Wi_stack))
        self.Wh_stack   = nn.Parameter(self.init_weights(self.shape_Wh_stack))
        self.bi_stack   = nn.Parameter(self.init_weights(self.shape_b_stack ))
        self.bh_stack   = nn.Parameter(self.init_weights(self.shape_b_stack ))

        self.sigmoid    = nn.Sigmoid()
        self.tanh       = nn.Tanh()

        self.lin = nn.Linear(hidden_size, input_size)
    
    def forward(self, x_inp, hs_prev = None, h_states = None, cs_prev = None):
        iters = x_inp.size(1)
        hs = self.hidden_size
        if hs_prev is None or h_states is None or cs_prev is None:
            hs_prev = self.init_states((hs, 1           ))
            cs_prev = self.init_states((hs, 1           ))
            h_states= self.init_states((hs, iters + 1   ))
        
        for i in range(iters):
            x_t     = x_inp[:,[i]]
            gates = self.Wi_stack @ x_t + self.bi_stack + self.Wh_stack @ hs_prev + self.bh_stack
            
            gate_input, gate_forget, gate_cell, gate_output = gates.chunk(4)

            gate_input     = self.sigmoid(gate_input)
            gate_forget    = self.sigmoid(gate_forget)
            gate_cell      = self.tanh(gate_cell)
            gate_output    = self.sigmoid(gate_output)

            state_cell      = gate_forget * cs_prev + gate_input * gate_cell
            state_hidden    = gate_output * self.tanh(gate_cell)
            h_states[:,[i+1]] = state_hidden    # store

            hs_prev = state_cell
            cs_prev = state_hidden

        output = self.lin(h_states[:,[-1]].T)

        return output, h_states, hs_prev, cs_prev
    
    