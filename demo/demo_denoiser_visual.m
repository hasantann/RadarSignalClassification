function demo_denoiser_visual()
% U-Net + ResNet-18 gurultu giderme aginin kullanimini gosteren demo.
%
% Veri setinden gurultulu/temiz bir spektrogram cifti secer, egitilmis
% denoiser'i gurultulu goruntuye uygular ve uc paneli yan yana cizer:
%   CLEAN (referans) | NOISY (gurultulu) | DENOISED (gurultu giderilmis)
% Her panel icin PSNR ve SSIM degerleri yazdirilir.
%
% Gerekli model: models/denoiser.mat
% Gerekli veri : dataRoot/specNoisyGray ve dataRoot/specCleanGray (eslesen adlar)

clear; clc; close all;

%% ===================== AYARLAR =====================
% Tum yollar bu betigin konumundan (repo koku) turetilir; makineye ozel
% mutlak yol yoktur.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
dataRoot    = fullfile(projectRoot, 'dataset', 'dataset_multiframe_patches');

noisyDir = fullfile(dataRoot, 'specNoisyGray');
cleanDir = fullfile(dataRoot, 'specCleanGray');

useGPU = canUseGPU;

% Denoiser norm modu (denoiser [0,1] giris/cikis ile egitildi):
%  "01" : giris [0,1]  / cikis [0,1]  (dogru mod; denoise_dataset.m ile ayni)
%  "m11": giris [-1,1] / cikis [-1,1]
normMode = "01";

%% ===================== DENOISER MODELİ =====================
denModelFile = fullfile(projectRoot, 'models', 'denoiser.mat');

assert(exist(denModelFile,'file')==2, 'Denoiser modeli bulunamadi: %s', denModelFile);
Sden   = load(denModelFile);
denNet = pick_net_any(Sden);

if useGPU && isa(denNet,'dlnetwork')
    denNet = dlupdate(@gpuArray, denNet);
end

[Hreq, Wreq, Creq] = get_net_input_size(denNet);
fprintf('Denoiser: %s\n', denModelFile);
fprintf('Net giris boyutu: [%d %d %d]\n', Hreq, Wreq, Creq);

%% ===================== ÖRNEK SEÇ =====================
files = dir(fullfile(noisyDir, '*.png'));
assert(~isempty(files), 'Gurultulu spektrogram bulunamadi: %s', noisyDir);

pick     = files(randi(numel(files)));
noisyImg = im2single(imread(fullfile(noisyDir, pick.name)));
cleanImg = im2single(imread(fullfile(cleanDir, pick.name)));

if ndims(noisyImg)==3, noisyImg = rgb2gray(noisyImg); end
if ndims(cleanImg)==3, cleanImg = rgb2gray(cleanImg); end

[H, W] = size(noisyImg);

%% ===================== DENOISER ÇIKIŞI =====================
denImg = run_denoiser(denNet, noisyImg, useGPU, normMode, [Hreq Wreq Creq], [H W]);

%% ===================== METRİKLER =====================
psnr_noisy = psnr(noisyImg, cleanImg);
ssim_noisy = ssim(noisyImg, cleanImg);
psnr_den   = psnr(denImg,   cleanImg);
ssim_den   = ssim(denImg,   cleanImg);

fprintf('\nOrnek: %s\n', pick.name);
fprintf('  Gurultulu vs Temiz : PSNR = %.2f dB, SSIM = %.3f\n', psnr_noisy, ssim_noisy);
fprintf('  Denoised  vs Temiz : PSNR = %.2f dB, SSIM = %.3f\n', psnr_den,   ssim_den);

%% ===================== GÖRSELLEŞTİRME =====================
figure('Name','U-Net + ResNet-18 Denoiser Demo', 'Position',[100 100 1500 500]);
tiledlayout(1,3,"Padding","compact","TileSpacing","compact");

nexttile;
imshow(cleanImg, []);
title('CLEAN (Referans)');
axis on; xlabel('Frekans'); ylabel('Zaman'); colormap gray;

nexttile;
imshow(noisyImg, []);
title(sprintf('NOISY\nPSNR=%.2f dB, SSIM=%.3f', psnr_noisy, ssim_noisy));
axis on; xlabel('Frekans'); ylabel('Zaman');

nexttile;
imshow(denImg, []);
title(sprintf('DENOISED\nPSNR=%.2f dB, SSIM=%.3f', psnr_den, ssim_den));
axis on; xlabel('Frekans'); ylabel('Zaman');

sgtitle(sprintf('best\\_unet\\_resnet18\\_tl | norm=%s | %s', normMode, pick.name), ...
    'Interpreter','tex');

end

%% ======================================================================
%  Denoiser'i tek bir gri goruntuye uygular (1/3 kanal ve norm modu esnek)
% ======================================================================
function out01 = run_denoiser(net, in01, useGPU, normMode, req, outHW)
Hreq = req(1); Wreq = req(2); Creq = req(3);

X = imresize(single(in01), [Hreq Wreq]);

if Creq==1
    Xin = X;
elseif Creq==3
    Xin = repmat(X,1,1,3);
else
    error('Desteklenmeyen giris kanal sayisi: %d', Creq);
end

if normMode=="m11"
    Xin = Xin*2 - 1;   % [0,1] -> [-1,1]
end

if isa(net,'dlnetwork')
    if ndims(Xin)==2
        dlX = dlarray(reshape(Xin,[Hreq Wreq 1 1]), 'SSCB');
    else
        dlX = dlarray(reshape(Xin,[Hreq Wreq 3 1]), 'SSCB');
    end
    if useGPU, dlX = gpuArray(dlX); end
    dlY = forward(net, dlX);
    Y   = gather(extractdata(dlY));
    Y   = squeeze(Y);
else
    if ndims(Xin)==2, Xin = reshape(Xin,[Hreq Wreq 1]); end
    Y = predict(net, Xin);
    Y = squeeze(Y);
end

if ndims(Y)==3 && size(Y,3)==3
    Y = rgb2gray(single(Y));
end
Y = single(Y);

if normMode=="m11"
    Y = (Y + 1)/2;     % [-1,1] -> [0,1]
end

Y = max(min(Y,1),0);
out01 = imresize(Y, outHW);
end

%% ======================================================================
%  MAT dosyasindan network degiskenini esnek sekilde secer
% ======================================================================
function net = pick_net_any(S)
cand = {'net','dlnet','denNet','bestNet'};
for i=1:numel(cand)
    if isfield(S,cand{i}), net = S.(cand{i}); return; end
end
fn = fieldnames(S);
for i=1:numel(fn)
    obj = S.(fn{i});
    if isa(obj,'dlnetwork') || isa(obj,'DAGNetwork') || isa(obj,'SeriesNetwork')
        net = obj; return;
    end
end
error('MAT dosyasinda network bulunamadi.');
end

function [H,W,C] = get_net_input_size(net)
L = net.Layers(1);
if isprop(L,'InputSize')
    sz = L.InputSize;
else
    sz = [224 224 1];
end
H = sz(1); W = sz(2);
if numel(sz)>=3, C = sz(3); else, C = 1; end
end
