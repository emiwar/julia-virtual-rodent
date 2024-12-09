using Distributions
import Plots

p = Plots.plot(grid=false, axis=true, size=(200, 800),
               xlim=(-3, 3), ylim=(0, 10))
for i=1:6
    mu = 0.5*randn()
    sigma = 0.2+rand()
    d = Normal(mu, sigma)
    h = 1.5(i-1)
    Plots.plot!(x->pdf(d, x)+h, label=false, linewidth=2.0)
    onesigma = mean(d) .+ std(d) .* [-1, 1]
    Plots.plot!(onesigma, pdf.(d, onesigma) .+ h,
                linestyle=:dash, color=:black, label=false)
    Plots.annotate!(mean(d), pdf(d, mean(d) + std(d))-0.2+h, "{\\sigma_$i}\\(t\\)")
    Plots.annotate!(mean(d), pdf(d, mean(d))+0.2+h, "{\\mu_$i}\\(t\\)")
end
p