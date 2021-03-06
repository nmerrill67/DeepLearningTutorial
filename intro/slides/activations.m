
sig = @(x) exp(x) ./ (exp(x) + 1);

x = -10:0.01:10;

figure

subplot(1,3,1)
plot(x, sig(x))
title("Sigmoid", 'fontsize', 20)
ylim([-1.5, 1.5])

subplot(1,3,2)
plot(x, tanh(x))
title("Tanh", 'fontsize', 20)
ylim([-1.5, 1.5])

subplot(1,3,3)
plot(x, relu(x))
title("ReLU", 'fontsize', 20)
ylim([-1.5, 1.5])
xlim([-10,10])


function y = relu(x)
    y = max(x, 0)
end