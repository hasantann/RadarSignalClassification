function train_denoiser_unet_resnet18
% Spektrogram gurultu giderme agi egitimi.
%
% Mimari : ResNet-18 tabanli enkoder + U-Net benzeri (skip baglantili) dekoder,
%          transfer ogrenme ile. Cikis regresyon katmanidir.
% Giris  : specNoisyGray (224x224 gri PNG), aga 224x224x3 olarak verilir.
% Hedef  : specCleanGray (224x224 gri PNG), 224x224x1 regresyon ciktisi.
%
% Egitim her epoch sonunda checkpoint kaydeder, en dusuk dogrulama kaybina
% sahip agi ayri saklar ve yarim kalan egitimi checkpoint'ten devam ettirir.
%
% Beklenen veri seti duzeni:
%   dataRoot/specNoisyGray/sample_000001.png
%   dataRoot/specCleanGray/sample_000001.png
%   dataRoot/labels.csv   (ilk sutun: dosya adi)

clc; clear;

%% ===================== PARAMETRELER =====================
inputSize   = [224 224 3];
miniBatch   = 8;
maxEpochs   = 40;
initialLR   = 1e-4;
valFraction = 0.2;

resumeFromCheckpoint  = true;
checkpointEveryEpoch  = 1;   % her epoch kaydet

%% ===================== DATASET YOLLARI =====================
% Tum yollar bu betigin konumundan (repo koku) turetilir; makineye ozel
% mutlak yol yoktur.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
dataRoot    = fullfile(projectRoot, 'dataset', 'dataset_multiframe_patches');

specNoisyRoot = fullfile(dataRoot, 'specNoisyGray');
specCleanRoot = fullfile(dataRoot, 'specCleanGray');
labelsPath    = fullfile(dataRoot, 'labels.csv');

modelsDir = fullfile(projectRoot, 'models');
ckptDir   = fullfile(projectRoot, 'checkpoints');
if ~exist(modelsDir,'dir'), mkdir(modelsDir); end
if ~exist(ckptDir,'dir'),   mkdir(ckptDir);   end

modelFile      = fullfile(modelsDir, 'denoiser.mat');
checkpointFile = fullfile(ckptDir,   'ckpt_denoiser.mat');

fprintf('\n=== ResNet18-UNet TL Denoiser (CKPT+BEST) ===\n');
fprintf('Data root: %s\n', dataRoot);

%% ===================== FILE LIST =====================
T = readtable(labelsPath);
names = string(T{:,1});
N = numel(names);

noisyPaths = fullfile(specNoisyRoot, names + ".png");
cleanPaths = fullfile(specCleanRoot, names + ".png");

imdsN = imageDatastore(noisyPaths, 'ReadSize', 1);
imdsC = imageDatastore(cleanPaths, 'ReadSize', 1);

%% ===================== TRAIN / VAL SPLIT =====================
rng(0);
idx = randperm(N);
nVal = round(valFraction*N);
valIdx = idx(1:nVal);
trIdx  = idx(nVal+1:end);

dsTrainRaw = combine(subset(imdsN,trIdx), subset(imdsC,trIdx));
dsValRaw   = combine(subset(imdsN,valIdx), subset(imdsC,valIdx));

dsTrain = transform(dsTrainRaw, @(d) preprocessPairGrayTo3ch(d, inputSize));
dsVal   = transform(dsValRaw,   @(d) preprocessPairGrayTo3ch(d, inputSize));

validationFrequency = max(1, floor(numel(trIdx)/miniBatch));

%% ===================== INIT / RESUME =====================
startEpoch = 1;
bestVal    = +inf;
bestNet    = [];
netLast    = [];

Tloss = [];
Vloss = [];

if resumeFromCheckpoint && exist(checkpointFile,'file')
    S = load(checkpointFile);
    fprintf('Checkpoint yüklendi: %s\n', checkpointFile);

    if isfield(S,'netLast'),     netLast     = S.netLast; end
    if isfield(S,'bestNet'),     bestNet     = S.bestNet; end
    if isfield(S,'bestVal'),     bestVal     = S.bestVal; end
    if isfield(S,'epoch'),       startEpoch  = S.epoch + 1; end
    if isfield(S,'Tloss'),       Tloss       = S.Tloss; end
    if isfield(S,'Vloss'),       Vloss       = S.Vloss; end
end

if isempty(netLast)
    fprintf('Sıfırdan model kuruluyor...\n');
    lgraph = build_resnet18_unet_denoiser(inputSize);
else
    fprintf('netLast üzerinden devam edilecek.\n');
    lgraph = layerGraph(netLast);
end

%% ===================== EPOCH LOOP =====================
for epoch = startEpoch:maxEpochs

    options = trainingOptions('adam', ...
        'InitialLearnRate',    initialLR, ...
        'MaxEpochs',           1, ...                     % 1 epoch/iter
        'MiniBatchSize',       miniBatch, ...
        'Shuffle',             'every-epoch', ...
        'ValidationData',      dsVal, ...
        'ValidationFrequency', validationFrequency, ...
        'Verbose',             false, ...
        'Plots',               'training-progress');

    % ---- 1 epoch train ----
    [netLast, info] = trainNetwork(dsTrain, lgraph, options);

    % Egitim kaybi (alan adi MATLAB surumune gore degisebildiginden esnek okunur)
    trainLoss = safeLast(info, {'TrainingLoss','TrainingRMSE','Loss'}, NaN);

    % Dogrulama kaybi; info icinde yoksa MSE olarak elle hesaplanir
    valLoss = safeLast(info, {'ValidationLoss','ValidationRMSE'}, NaN);
    if ~isfinite(valLoss)
        valLoss = computeValMSE(netLast, dsVal, miniBatch);
    end

    Tloss(end+1) = double(trainLoss);
    Vloss(end+1) = double(valLoss);

    % ---- best net ----
    if valLoss < bestVal
        bestVal = valLoss;
        bestNet = netLast;
        save(modelFile,'bestNet','bestVal','Tloss','Vloss','-v7.3');
        fprintf('En iyi model guncellendi. bestVal=%.6f\n', bestVal);
    end

    % ---- checkpoint ----
    if mod(epoch, checkpointEveryEpoch) == 0
        save(checkpointFile, ...
            'netLast','bestNet','bestVal','epoch','Tloss','Vloss', ...
            'inputSize','miniBatch','maxEpochs','initialLR','valFraction', ...
            '-v7.3');
        fprintf('Checkpoint kaydedildi (epoch %d/%d)\n', epoch, maxEpochs);
    end

    fprintf('Epoch %3d/%3d | TrainLoss %.6f | ValLoss %.6f | Best %.6f\n', ...
        epoch, maxEpochs, trainLoss, valLoss, bestVal);

    % next epoch starts from last net
    lgraph = layerGraph(netLast);
end

fprintf('\nEgitim tamamlandi.\nEn iyi model: %s\nCheckpoint: %s\n', modelFile, checkpointFile);

end

%% =====================================================================
function dataOut = preprocessPairGrayTo3ch(data, inputSize)
% data: {noisyImage, cleanImage}
% noisy  -> 224x224x3  (0..1)
% clean  -> 224x224x1  (0..1)

noisy = data{1};
clean = data{2};

if ndims(noisy) == 3, noisy = rgb2gray(noisy); end
if ndims(clean) == 3, clean = rgb2gray(clean); end

noisy = im2single(noisy);
clean = im2single(clean);

H = inputSize(1); W = inputSize(2);

if ~isequal(size(noisy), [H W]), noisy = imresize(noisy, [H W]); end
if ~isequal(size(clean), [H W]), clean = imresize(clean, [H W]); end

noisy3 = repmat(noisy, 1, 1, 3);
clean1 = reshape(clean, H, W, 1);

dataOut = {noisy3, clean1};
end

function lgraph = build_resnet18_unet_denoiser(inputSize)
% ResNet18 encoder + UNet-like decoder (skip)
% Robust: layer names differ by MATLAB version. We auto-pick existing names.

net = resnet18;
lgraph = layerGraph(net);

% ---- Replace input ----
lgraph = replaceLayer(lgraph,'data', ...
    imageInputLayer(inputSize,'Name','data','Normalization','zerocenter'));

% ---- Remove classifier head ----
layersToRemove = {'pool5','fc1000','prob','ClassificationLayer_predictions'};
lgraph = removeLayers(lgraph, layersToRemove);

% ---- Available names ----
names = string({lgraph.Layers.Name});
skip1 = pickFirstExisting(names, ["conv1_relu"]);
skip2 = pickFirstExisting(names, ["res2b_relu"]);
skip3 = pickFirstExisting(names, ["res3b_relu"]);
skip4 = pickFirstExisting(names, ["res4b_relu"]);
bott  = pickFirstExisting(names, ["res5b_relu"]);

% ---- Decoder sequence (MATLAB auto-connects sequential layers) ----
decoder = [
    transposedConv2dLayer(4,256,'Stride',2,'Cropping','same','Name','up1')
    reluLayer('Name','up1_relu')
    depthConcatenationLayer(2,'Name','cat1')
    convolution2dLayer(3,256,'Padding','same','Name','dec1_conv')
    reluLayer('Name','dec1_relu')

    transposedConv2dLayer(4,128,'Stride',2,'Cropping','same','Name','up2')
    reluLayer('Name','up2_relu')
    depthConcatenationLayer(2,'Name','cat2')
    convolution2dLayer(3,128,'Padding','same','Name','dec2_conv')
    reluLayer('Name','dec2_relu')

    transposedConv2dLayer(4,64,'Stride',2,'Cropping','same','Name','up3')
    reluLayer('Name','up3_relu')
    depthConcatenationLayer(2,'Name','cat3')
    convolution2dLayer(3,64,'Padding','same','Name','dec3_conv')
    reluLayer('Name','dec3_relu')

    transposedConv2dLayer(4,32,'Stride',2,'Cropping','same','Name','up4')
    reluLayer('Name','up4_relu')
    depthConcatenationLayer(2,'Name','cat4')
    convolution2dLayer(3,32,'Padding','same','Name','dec4_conv')
    reluLayer('Name','dec4_relu')

    transposedConv2dLayer(4,16,'Stride',2,'Cropping','same','Name','up5')
    reluLayer('Name','up5_relu')

    convolution2dLayer(1,1,'Padding','same','Name','out_conv')
    regressionLayer('Name','regressionoutput')
];

lgraph = addLayers(lgraph, decoder);

% ---- Manual connections needed ----
% bottleneck -> decoder start
lgraph = connectLayers(lgraph, bott, 'up1');

% IMPORTANT: DO NOT connect cat*/in1 (auto-connected by sequence)
% Only connect skip -> cat*/in2
lgraph = connectLayers(lgraph, skip4, 'cat1/in2');
lgraph = connectLayers(lgraph, skip3, 'cat2/in2');
lgraph = connectLayers(lgraph, skip2, 'cat3/in2');
lgraph = connectLayers(lgraph, skip1, 'cat4/in2');

% ---- Optional: lower LR factors (conv + tconv only) ----
L = lgraph.Layers;
for i = 1:numel(L)
    if isa(L(i),'nnet.cnn.layer.Convolution2DLayer') || ...
       isa(L(i),'nnet.cnn.layer.TransposedConvolution2DLayer')
        L(i).WeightLearnRateFactor = 0.2;
        L(i).BiasLearnRateFactor   = 0.2;
    end
end

% Rebuild graph while preserving connections
conns = lgraph.Connections;
lgraph = layerGraph();
for i = 1:numel(L)
    lgraph = addLayers(lgraph, L(i));
end
for c = 1:size(conns,1)
    lgraph = connectLayers(lgraph, conns.Source{c}, conns.Destination{c});
end

end

% ---------------- helpers ----------------
function name = pickFirstExisting(allNames, candidates)
idx = find(ismember(allNames, candidates), 1, 'first');
if isempty(idx)
    % Skip katmani bulunamadiysa mevcut katman adlarindan ornek gosterilir
    error("Skip layer bulunamadi. Adaylar: %s\nMevcut layer isimlerinden ornek (ilk 40):\n%s", ...
        strjoin(candidates,", "), strjoin(allNames(1:min(40,numel(allNames))), ", "));
end
name = allNames(idx);
end

%% =====================================================================
function v = safeLast(info, fieldCandidates, defaultVal)
v = defaultVal;
for k = 1:numel(fieldCandidates)
    f = fieldCandidates{k};
    if isprop(info,f) || isfield(info,f)
        try
            vv = info.(f);
            if ~isempty(vv), v = vv(end); return; end
        catch
        end
    end
end
end

%% =====================================================================
function mseVal = computeValMSE(net, dsVal, miniBatch)
% Manual validation MSE if info.ValidationLoss is missing
reset(dsVal);
mseSum = 0;
count  = 0;

while hasdata(dsVal)
    batch = read(dsVal);
    % batch is {X, T}
    X = batch{1};
    T = batch{2};

    % ensure batch dims (some datastores return single sample only)
    if ndims(X)==3 && size(X,3)==3
        Xb = reshape(X, size(X,1), size(X,2), size(X,3), 1);
    else
        Xb = X;
    end
    if ndims(T)==2
        Tb = reshape(T, size(T,1), size(T,2), 1, 1);
    else
        Tb = T;
    end

    Y = predict(net, Xb);
    Y = squeeze(Y(:,:,1,:));
    Tb = squeeze(Tb(:,:,1,:));

    err = (single(Y) - single(Tb)).^2;
    mseSum = mseSum + mean(err(:));
    count  = count + 1;

    % küçük batch simülasyonu: dsVal zaten tek tek okuyor olabilir, sorun değil
end

mseVal = mseSum / max(count,1);
end
