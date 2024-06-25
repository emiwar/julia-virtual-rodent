import Plots

p = Plots.plot(xlim=(0, 1000))
for i in 1:3
    Plots.plot!(p, test_env.com_targets[i, :], label="xyz"[i:i])
end
p
#Plots.show(p)