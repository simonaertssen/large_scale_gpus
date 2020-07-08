clear all; close all; clc
%%
addpath('Benchmark ADM_Direct/');
addpath('Benchmark ADM_DirectPeer P9/');
addpath('Benchmark CublasXt/');
addpath('Optimal Block Dimension CublasXt/');

set(groot,'DefaultAxesXGrid','on')
set(groot,'DefaultAxesYGrid','on')
rect = [50 800 1000 400];
cm = [1,0,0; 0, 0.7410, 0.4470; 0, 0.4470, 0.6410];

Ns = [16384, 32768];
colors = ['#0072BD','#D95319'];

%% Optimal cublasXt dimensions: P9SH

filename = 'Optimal Block Dimension CublasXt/FinalFinalresults_optimalBlockdimCublasXt_PS9.txt';

close all
f1 = figure('Renderer', 'painters', 'Position', [50 800 1000 300]);

results = table2array(readtable(filename));

for j = 1:2
imcol(j) = subplot(1,2,j);

n = results(:, 1);
idx = n == Ns(j);
n = n(idx);
blockdim = results(:, 2);
blockdim = blockdim(idx);
start = find(idx == 1, 1, 'first')
l = length(n);

if j == 2
    start = start/3
end 
GFLOPS = results(start:start+l, 3)
mGFLOPS = zeros(l/3, 1);
vGFLOPS = zeros(l/3, 1);
for i = 1:l/3
    mGFLOPS(i) = mean(GFLOPS(i:(i+3)));
    vGFLOPS(i) = std(GFLOPS(i:(i+3)));
end
%mGFLOPS = accumarray(ceil((1:numel(GFLOPS))'/3),GFLOPS(:),[],@mean);
a = tinv(0.95,2)*vGFLOPS/sqrt(3);

hold on; box on;
counter = [blockdim(1:3:end); flipud(blockdim(1:3:end))];
inBetween = [mGFLOPS + a; flipud(max(0, mGFLOPS - a))];
h = fill(counter, inBetween, 'b');
set(h,'facealpha', .1, 'FaceColor', '#0072BD','LineStyle','none')
plot(blockdim(1:3:end), mGFLOPS, '-o', 'LineWidth', 2, 'color', cm(3,:));
plot(blockdim(1:3:end), mGFLOPS + a, ':k', 'LineWidth', 1);
plot(blockdim(1:3:end), max(0, mGFLOPS - a), ':k', 'LineWidth', 1);

title(['\fontname{SansSerif}',' N = ',num2str(Ns(j))], 'FontSize', 24);
xlabel('\fontname{SansSerif}Block dimension T', 'FontSize', 16)
ylabel('\fontname{SansSerif}Performance [GFLOPS]', 'FontSize', 16)
xticks(1024:1024:14336)
xtickangle(45)

a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','SansSerif','fontsize',12)

end

print(f1, 'Optimal Block Dimension CublasXt/optimalBlockdimCublasXtPS9.png', '-dpng', '-r400')





