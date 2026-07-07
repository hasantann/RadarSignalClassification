function denoise_dataset()
% Gurultu giderilmis veri seti uretimi.
%
% Egitilmis denoiser agini (train_denoiser_unet_resnet18.m ciktisi) tum
% gurultulu gri spektrogramlara (specNoisyGray) uygular ve sonuclari
% specDenoisedGray klasorune kaydeder. Bu klasor daha sonra
% train_classifier_denoised.m ile siniflandirici egitiminde kullanilir.
%
% Klasor agaci korunur; zaten uretilmis ciktilar atlanir.

clc; clear;

%% === PATHS ===
% Tum yollar bu betigin konumundan (repo koku) turetilir; makineye ozel
% mutlak yol yoktur.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
dataRoot    = fullfile(projectRoot, 'dataset', 'dataset_multiframe_patches');
inRoot      = fullfile(dataRoot, 'specNoisyGray');
outRoot     = fullfile(dataRoot, 'specDenoisedGray');

if ~exist(outRoot,'dir'); mkdir(outRoot); end

%% === DENOISER MODEL ===
% Model dosyasi net / bestNet / denNet degiskenlerinden birini icerebilir.
modelFile = fullfile(projectRoot,'models','denoiser.mat');

S = load(modelFile);

if isfield(S,'net')
    denNet = S.net;
elseif isfield(S,'bestNet')
    denNet = S.bestNet;
elseif isfield(S,'denNet')
    denNet = S.denNet;
else
    error('Model dosyasinda net/bestNet/denNet bulunamadi: %s', modelFile);
end

useGPU = canUseGPU;

% Normalizasyon modu:
%  - "minus1to1": x = (x-0.5)/0.5
%  - "zero_one" : x = x
normMode = "zero_one";

% Çıktıyı kaydetmeden önce ters-normalize:
%  - "minus1to1": y = (y*0.5+0.5)
%  - "zero_one" : y = y
outMode  = normMode;

%% === NET INPUT SIZE (auto) ===
[inH, inW, inC] = inferInputSize(denNet);

fprintf('Input root : %s\n', inRoot);
fprintf('Output root: %s\n', outRoot);
fprintf('Model      : %s\n', modelFile);
fprintf('Net input  : %dx%dx%d\n', inH, inW, inC);
fprintf('Device     : %s\n', ternary(useGPU,'GPU','CPU'));
fprintf('NormMode   : %s\n\n', normMode);

%% === LIST FILES (recursive) ===
files = dir(fullfile(inRoot, '**', '*.png'));
if isempty(files)
    error('PNG bulunamadi: %s', inRoot);
end

N = numel(files);
fprintf('Toplam PNG: %d\n', N);

%% === PROCESS LOOP ===
t0 = tic;
for k = 1:N
    inPath = fullfile(files(k).folder, files(k).name);

    % relative path to preserve folder tree
    relFolder = erase(files(k).folder, inRoot);
    if startsWith(relFolder, filesep), relFolder = relFolder(2:end); end

    outDir = fullfile(outRoot, relFolder);
    if ~exist(outDir,'dir'); mkdir(outDir); end

    outPath = fullfile(outDir, files(k).name);

    % skip if exists
    if exist(outPath,'file')
        continue;
    end

    % ---- read & ensure grayscale ----
    I = imread(inPath);
    if ndims(I) == 3
        I = rgb2gray(I);
    end

    % ---- to single [0,1] ----
    Is = im2single(I);

    % ---- resize to net input ----
    Isr = imresize(Is, [inH inW], 'bilinear');

    % ---- channel adapt ----
    if inC == 3
        Isr = repmat(Isr, 1, 1, 3);
    elseif inC == 1
        % ok
    else
        % nadiren olur: ilk kanalı al
        if size(Isr,3) > inC
            Isr = Isr(:,:,1:inC);
        else
            Isr = repmat(Isr,1,1,inC);
        end
    end

    % ---- normalize input ----
    X = normalizeInput(Isr, normMode);

    % ---- denoise ----
    Y = runDenoiser(denNet, X, useGPU);

    % ---- output inverse-normalize + clamp ----
    Y = inverseNormalizeOutput(Y, outMode);
    Y = max(min(Y,1),0);

    % ---- if net output is 3ch, convert to gray ----
    if ndims(Y) == 3 && size(Y,3) == 3
        Y = rgb2gray(Y);
    end

    % ---- resize back to original size (so downstream labels match) ----
    Y = imresize(Y, [size(I,1) size(I,2)], 'bilinear');

    % ---- save uint8 png ----
    imwrite(im2uint8(Y), outPath);

    if mod(k,100)==0 || k==N
        fprintf('%6d/%6d done | last: %s\n', k, N, files(k).name);
    end
end

fprintf('\nTamamlandi. Sure: %.1f sn\n', toc(t0));
end

%% =================== HELPERS ===================

function [H,W,C] = inferInputSize(net)
% Supports dlnetwork / DAGNetwork / SeriesNetwork
try
    L = net.Layers(1);
    if isprop(L,'InputSize')
        sz = L.InputSize;
        H = sz(1); W = sz(2); C = sz(3);
        return;
    end
catch
end

% dlnetwork can store input sizes differently; fallback:
try
    % Common case: find imageInputLayer by type/name
    layers = net.Layers;
    idx = find(arrayfun(@(x) isa(x,'nnet.cnn.layer.ImageInputLayer'), layers), 1);
    if ~isempty(idx)
        sz = layers(idx).InputSize;
        H = sz(1); W = sz(2); C = sz(3);
        return;
    end
end

% Last resort:
H = 224; W = 224; C = 1;
warning('Input size bulunamadi, default %dx%dx%d kullanildi.', H,W,C);
end

function Xn = normalizeInput(X, mode)
switch string(mode)
    case "minus1to1"
        Xn = (single(X) - 0.5) / 0.5;   % [0,1] -> [-1,1]
    case "zero_one"
        Xn = single(X);                 % [0,1]
    otherwise
        error('Bilinmeyen normMode: %s', mode);
end
end

function Y = inverseNormalizeOutput(Y, mode)
Y = single(Y);
switch string(mode)
    case "minus1to1"
        Y = Y*0.5 + 0.5;               % [-1,1] -> [0,1]
    case "zero_one"
        % do nothing
    otherwise
        error('Bilinmeyen outMode: %s', mode);
end
end

function Y = runDenoiser(net, X, useGPU)
% X: HxWxC single, range depends on normMode
% output Y: HxWxC single
if useGPU
    X = gpuArray(X);
end

% dlnetwork ise dlarray ile forward
if isa(net,'dlnetwork')
    dlX = dlarray(X, 'SSC');
    dlY = forward(net, dlX);
    Y   = gather(extractdata(dlY));
else
    % DAGNetwork/SeriesNetwork -> predict
    Y = predict(net, X);
    if useGPU
        Y = gather(Y);
    end
end
end

function out = ternary(cond,a,b)
if cond, out = a; else, out = b; end
end
