
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

imcol(1) = subplot(1,2,1);

results = table2array(readtable(filename));

n = results(:, 1);
idx = n == Ns(1);
l = length(n(idx));
results16 = results(idx,:);
blockdim = results16(:, 2);

GFLOPS = results16(:, 3);
mGFLOPS = zeros(l/3, 1);
vGFLOPS = zeros(l/3, 1);
for i = 1:l/3
    val = 1 + (i-1)*3;    % val = 1, 4, 7, 10 ... 
    mGFLOPS(i) = mean(GFLOPS(val:(val+2)));
    vGFLOPS(i) = std(GFLOPS(val:(val+2)));
%     mGFLOPS(i) = mean(GFLOPS(i:(i+3)));
%     vGFLOPS(i) = std(GFLOPS(i:(i+3)));
end
a = tinv(0.95,2)*vGFLOPS/sqrt(3);

hold on; box on;
counter = [blockdim(1:3:end); flipud(blockdim(1:3:end))];
inBetween = [mGFLOPS + a; flipud(max(0, mGFLOPS - a))];
h = fill(counter, inBetween, 'b');
set(h,'facealpha', .1, 'FaceColor', '#0072BD','LineStyle','none')
plot(blockdim(1:3:end), mGFLOPS, '-o', 'LineWidth', 2, 'color', cm(3,:));
plot(blockdim(1:3:end), mGFLOPS + a, ':k', 'LineWidth', 1);
plot(blockdim(1:3:end), max(0, mGFLOPS - a), ':k', 'LineWidth', 1);

title(['\fontname{SansSerif}',' N = ',num2str(Ns(1))], 'FontSize', 24);
xlabel('\fontname{SansSerif}Tile dimension T', 'FontSize', 16)
ylabel('\fontname{SansSerif}Performance [GFLOPS]', 'FontSize', 16)
xticks(1024:1024:14336)
xtickangle(45)
ylim([0, 15000])
yticks(0:3000:15000)

a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','SansSerif','fontsize',12)

% Right plot:
imcol(2) = subplot(1,2,2);
results = table2array(readtable(filename));

n = results(:, 1);
idx = n == Ns(2);
l = length(n(idx));

results32 = results(idx,:);
blockdim = results32(:, 2);

GFLOPS = results32(:, 3);
mGFLOPS = zeros(l/3, 1);
vGFLOPS = zeros(l/3, 1);
for i = 1:l/3
    val = 1 + (i-1)*3;    % val = 1, 4, 7, 10 ... 
    mGFLOPS(i) = mean(GFLOPS(val:(val+2)));
    vGFLOPS(i) = std(GFLOPS(val:(val+2)));
%     mGFLOPS(i) = mean(GFLOPS(i:(i+3)));
%     vGFLOPS(i) = std(GFLOPS(i:(i+3)));
end
a = tinv(0.95,2)*vGFLOPS/sqrt(3);

hold on; box on;
counter = [blockdim(1:3:end); flipud(blockdim(1:3:end))];
inBetween = [mGFLOPS + a; flipud(max(0, mGFLOPS - a))];
h = fill(counter, inBetween, 'b');
set(h,'facealpha', .1, 'FaceColor', '#0072BD','LineStyle','none')
plot(blockdim(1:3:end), mGFLOPS, '-o', 'LineWidth', 2, 'color', cm(3,:));
plot(blockdim(1:3:end), mGFLOPS + a, ':k', 'LineWidth', 1);
plot(blockdim(1:3:end), max(0, mGFLOPS - a), ':k', 'LineWidth', 1);

title(['\fontname{SansSerif}',' N = ',num2str(Ns(2))], 'FontSize', 24);
xlabel('\fontname{SansSerif}Tile dimension T', 'FontSize', 16)
ylabel('\fontname{SansSerif}Performance [GFLOPS]', 'FontSize', 16)
xticks(blockdim(1:3:end))
xlim([0, blockdim(end) + 1024])

ylim([0, 15000])
yticks(0:3000:15000)

xtickangle(45)

a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','SansSerif','fontsize',12)

print(f1, 'Optimal Block Dimension CublasXt/optimalBlockdimCublasXtPS9.png', '-dpng', '-r400')


%% Optimal cublasXt dimensions: SXM2SH
filename = 'Optimal Block Dimension CublasXt/FinalResults_optimalBlockdimCublasXtSXM2SH.txt';

close all
f2 = figure('Renderer', 'painters', 'Position', [50 800 1000 300]);

imcol(1) = subplot(1,2,1);

results = table2array(readtable(filename));

n = results(:, 1);
idx = n == Ns(1);
l = length(n(idx));
results16 = results(idx,:);
blockdim = results16(:, 2);

GFLOPS = results16(:, 3);
mGFLOPS = zeros(l/3, 1);
vGFLOPS = zeros(l/3, 1);
for i = 1:l/3
    val = 1 + (i-1)*3;    % val = 1, 4, 7, 10 ... 
    mGFLOPS(i) = mean(GFLOPS(val:(val+2)));
    vGFLOPS(i) = std(GFLOPS(val:(val+2)));
%     mGFLOPS(i) = mean(GFLOPS(i:(i+3)));
%     vGFLOPS(i) = std(GFLOPS(i:(i+3)));
end
a = tinv(0.05,2)*vGFLOPS/sqrt(3);

hold on; box on;
counter = [blockdim(1:3:end); flipud(blockdim(1:3:end))];
inBetween = [mGFLOPS + a; flipud(max(0, mGFLOPS - a))];
h = fill(counter, inBetween, 'b');
set(h,'facealpha', .1, 'FaceColor', 'r','LineStyle','none')
plot(blockdim(1:3:end), mGFLOPS, '-or', 'LineWidth', 2);
plot(blockdim(1:3:end), mGFLOPS + a, ':k', 'LineWidth', 1);
plot(blockdim(1:3:end), max(0, mGFLOPS - a), ':k', 'LineWidth', 1);

title(['\fontname{SansSerif}',' N = ',num2str(Ns(1))], 'FontSize', 24);
xlabel('\fontname{SansSerif}Tile dimension T', 'FontSize', 16)
ylabel('\fontname{SansSerif}Performance [GFLOPS]', 'FontSize', 16)
xticks(1024:1024:14336)
xtickangle(45)
ylim([0, 10000])
yticks(0:1000:10000)

a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','SansSerif','fontsize',12)

% Right plot:
imcol(2) = subplot(1,2,2);
results = table2array(readtable(filename));

n = results(:, 1);
idx = n == Ns(2);
l = length(n(idx));

results32 = results(idx,:);
blockdim = results32(:, 2);

GFLOPS = results32(:, 3);
mGFLOPS = zeros(l/3, 1);
vGFLOPS = zeros(l/3, 1);
for i = 1:l/3
    val = 1 + (i-1)*3;    % val = 1, 4, 7, 10 ... 
    mGFLOPS(i) = mean(GFLOPS(val:(val+2)));
    vGFLOPS(i) = std(GFLOPS(val:(val+2)));
%     mGFLOPS(i) = mean(GFLOPS(i:(i+3)));
%     vGFLOPS(i) = std(GFLOPS(i:(i+3)));
end
a = tinv(0.05,2)*vGFLOPS/sqrt(3);

hold on; box on;
counter = [blockdim(1:3:end); flipud(blockdim(1:3:end))];
inBetween = [mGFLOPS + a; flipud(max(0, mGFLOPS - a))];
h = fill(counter, inBetween, 'b');
set(h,'facealpha', .1, 'FaceColor', 'r','LineStyle','none')
plot(blockdim(1:3:end), mGFLOPS, '-or', 'LineWidth', 2);
plot(blockdim(1:3:end), mGFLOPS + a, ':k', 'LineWidth', 1);
plot(blockdim(1:3:end), max(0, mGFLOPS - a), ':k', 'LineWidth', 1);

title(['\fontname{SansSerif}',' N = ',num2str(Ns(2))], 'FontSize', 24);
xlabel('\fontname{SansSerif}Tile dimension T', 'FontSize', 16)
ylabel('\fontname{SansSerif}Performance [GFLOPS]', 'FontSize', 16)
xticks(blockdim(1:3:end))
xlim([0, blockdim(end) + 1024])
xtickangle(45)
ylim([0, 10000])
yticks(0:2000:10000)

a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'FontName','SansSerif','fontsize',12)

print(f2, 'Optimal Block Dimension CublasXt/FinalResults_optimalBlockdimCublasXtSXM2SH.png', '-dpng', '-r400')


%% Benchmark cublasXt: SXM2SH
filename = 'Benchmark CublasXt/Finalresults_benchmarkCublasXtSXM2SH.txt';

close all
f3 = figure('Renderer', 'painters', 'Position', [50 800 500 300]);

results = table2array(readtable(filename));
n = results(:, 1);
l = length(n);
blockdim = results(:, 2);

h = axes;
set(h,'xscale','log')

GFLOPS = results(:, 3);
mGFLOPS = zeros(l/3, 1);
vGFLOPS = zeros(l/3, 1);
for i = 1:l/3
    val = 1 + (i-1)*3;    % val = 1, 4, 7, 10 ... 
    mGFLOPS(i) = mean(GFLOPS(val:(val+2)));
    vGFLOPS(i) = std(GFLOPS(val:(val+2)));
%     mGFLOPS(i) = mean(GFLOPS(i:(i+3)));
%     vGFLOPS(i) = std(GFLOPS(i:(i+3)));
end
a = tinv(0.95,2)*vGFLOPS/sqrt(3);

hold on; box on;
counter = [n(1:3:end); flipud(n(1:3:end))];
inBetween = [mGFLOPS + a; flipud(max(0, mGFLOPS - a))];
h = fill(counter, inBetween, 'b');
set(h,'facealpha', .1, 'FaceColor', 'r','LineStyle','none')
perf = plot(n(1:3:end), mGFLOPS, '-or', 'LineWidth', 2);
plot(n(1:3:end), mGFLOPS + a, ':k', 'LineWidth', 1);
plot(n(1:3:end), max(0, mGFLOPS - a), ':k', 'LineWidth', 1);

%title(['\fontname{SansSerif}',' N = ',num2str(Ns(1))], 'FontSize', 24);
xlabel('\fontname{SansSerif}Matrix dimension N', 'FontSize', 16)
ylabel('\fontname{SansSerif}Performance [GFLOPS]', 'FontSize', 16)
xlim([0, n(end) + 1024])

xticks(n(1:3:end))
xtickangle(45)
ylim([0, 14000])
yticks(0:2000:14000)

ax = get(gca,'XTickLabel');
set(gca,'XTickLabel',ax,'FontName','SansSerif','fontsize',12)

yyaxis right
tiledim = plot(n(1:3:end), blockdim(1:3:end), '-xk', 'LineWidth', 2);
ylim([0, 10000])
yticks(0:1024:8192)
ylabel('\fontname{SansSerif}Tile dimension T', 'FontSize', 16)
ax = gca;
ax.YColor = 'k';

legend([perf; tiledim], 'Performance', 'Tile dimension', 'Location', 'northwest');
print(f3, 'Benchmark CublasXt/Finalresults_benchmarkCublasXtSXM2SH.png', '-dpng', '-r400')

%% Benchmark cublasXt: P9
filename = 'Benchmark CublasXt/Finalresults_benchmarkCublasXtP9SH.txt';

close all
f4 = figure('Renderer', 'painters', 'Position', [50 800 500 300]);

results = table2array(readtable(filename));
n = results(:, 1);
l = length(n);
blockdim = results(:, 2);

h = axes;
set(h,'xscale','log')

GFLOPS = results(:, 3);
mGFLOPS = zeros(l/3, 1);
vGFLOPS = zeros(l/3, 1);
for i = 1:l/3
    val = 1 + (i-1)*3;    % val = 1, 4, 7, 10 ... 
    mGFLOPS(i) = mean(GFLOPS(val:(val+2)));
    vGFLOPS(i) = std(GFLOPS(val:(val+2)));
%     mGFLOPS(i) = mean(GFLOPS(i:(i+3)));
%     vGFLOPS(i) = std(GFLOPS(i:(i+3)));
end
a = tinv(0.95,2)*vGFLOPS/sqrt(3);

hold on; box on;
counter = [n(1:3:end); flipud(n(1:3:end))];
inBetween = [mGFLOPS + a; flipud(max(0, mGFLOPS - a))];
h = fill(counter, inBetween, 'b');
set(h,'facealpha', .1, 'FaceColor', '#0072BD','LineStyle','none')
perf = plot(n(1:3:end), mGFLOPS, '-or', 'LineWidth', 2, 'color', '#0072BD');
plot(n(1:3:end), mGFLOPS + a, ':k', 'LineWidth', 1);
plot(n(1:3:end), max(0, mGFLOPS - a), ':k', 'LineWidth', 1);

%title(['\fontname{SansSerif}',' N = ',num2str(Ns(1))], 'FontSize', 24);
xlabel('\fontname{SansSerif}Matrix dimension N', 'FontSize', 16)
ylabel('\fontname{SansSerif}Performance [GFLOPS]', 'FontSize', 16)
xticks(n(1:3:end))
xlim([0, n(end) + 1024])

xtickangle(45)
ylim([0, 14000])
yticks(0:2000:14000)

ax = get(gca,'XTickLabel');
set(gca,'XTickLabel',ax,'FontName','SansSerif','fontsize',12)

yyaxis right
tiledim = plot(n(1:3:end), blockdim(1:3:end), '-xk', 'LineWidth', 1);
ylim([0, 10000])
yticks(0:1024:8192)
ylabel('\fontname{SansSerif}Tile dimension T', 'FontSize', 16)
ax = gca;
ax.YColor = 'k';

legend([perf; tiledim], 'Performance', 'Tile dimension', 'Location', 'northwest');

print(f4, 'Benchmark CublasXt/Finalresults_benchmarkCublasXtP9SH.png', '-dpng', '-r400')


%% Benchmark ADM: SXM2SH
filename = 'Benchmark ADM_Direct/Finalresults_benchmarkADM_DirectSXM2SH.txt';

close all
f5 = figure('Renderer', 'painters', 'Position', [50 800 500 300]);

results = table2array(readtable(filename));
n = results(:, 1);
l = length(n);
blockdim = results(:, 2);

h = axes;
set(h,'xscale','log')

GFLOPS = results(:, 3);
mGFLOPS = zeros(l/3, 1);
vGFLOPS = zeros(l/3, 1);
for i = 1:l/3
    val = 1 + (i-1)*3;    % val = 1, 4, 7, 10 ... 
    mGFLOPS(i) = mean(GFLOPS(val:(val+2)));
    vGFLOPS(i) = std(GFLOPS(val:(val+2)));
%     mGFLOPS(i) = mean(GFLOPS(i:(i+3)));
%     vGFLOPS(i) = std(GFLOPS(i:(i+3)));
end
a = tinv(0.95,2)*vGFLOPS/sqrt(3);

hold on; box on;
counter = [n(1:3:end); flipud(n(1:3:end))];
inBetween = [mGFLOPS + a; flipud(max(0, mGFLOPS - a))];
h = fill(counter, inBetween, 'b');
set(h,'facealpha', .1, 'FaceColor', 'r','LineStyle','none')
plot(n(1:3:end), mGFLOPS, '-or', 'LineWidth', 2);
plot(n(1:3:end), mGFLOPS + a, ':k', 'LineWidth', 1);
plot(n(1:3:end), max(0, mGFLOPS - a), ':k', 'LineWidth', 1);

%title(['\fontname{SansSerif}',' N = ',num2str(Ns(1))], 'FontSize', 24);
xlabel('\fontname{SansSerif}Tile dimension T', 'FontSize', 16)
ylabel('\fontname{SansSerif}Performance [GFLOPS]', 'FontSize', 16)
xticks(n(1:3:end))
xtickangle(45)

ax = get(gca,'XTickLabel');
set(gca,'XTickLabel',ax,'FontName','SansSerif','fontsize',12)


print(f5, 'Benchmark ADM_Direct/Finalresults_benchmarkADM_DirectSXM2SH.png', '-dpng', '-r400')

