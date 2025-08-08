vocab_size = 1
d_model = 128
num_layers = 1
d_ff = 4*d_model
batch_size = 4
seq_len = 16384
P = 2*vocab_size*d_model + num_layers*(4*d_model*d_model + 3*d_model*d_ff)
total = P * 2 + num_layers * batch_size * seq_len * d_model * 4
gb = total / 1073741824
print(gb)