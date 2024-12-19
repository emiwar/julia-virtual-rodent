import Flux

lstmcell = Flux.LSTMCell(5, 2)

x0 = randn(5, 10)
h0 = lstmcell.state0
h1, y1 = lstmcell(h0, x0)

h = h1
h, y = lstmcell(h, x0)

chain = (Flux.LSTMCell(5, 2), Flux.LSTMCell(2, 3))
x0 = randn(5, 10)
h0 = map(x->x.state0, chain)

accumulate((cell, (h, x))->cell(h, x), chain, init=(h0, x0))




lstm = Flux.LSTM(5, 2)
@code_warntype lstm(x0)

r2 = Flux.Recur((Flux.LSTMCell(5, 2), Flux.LSTMCell(2, 3)))
r2(x0)



c2 = Flux.Chain(Flux.LSTM(5=>13), Flux.LSTM(13=>2))
y0 = c2(x0)
h1 = map(l->l.state, c2)
for i=1:5
    c2(x0)
end
h6 = map(l->l.state, c2)

for (cell, hidden) in zip(c2, h1)
    cell.state = hidden
end
for i=1:5
    c2(x0)
end
h6_again = map(l->l.state, c2)
