function train_classifier_noisy
% Cok etiketli radar sinyal siniflandirici egitimi (GURULTULU spektrogramlar).
%
% Mimari : ImageNet on-egitimli GoogLeNet, cok etiketli cikis icin sigmoid +
%          agirlikli BCE kaybi ile uyarlanmistir.
% Giris  : Gurultu giderme uygulanmamis gri spektrogram yamalarindan
%          (specGray) olusan yamalar; her yama 224x224x3 olarak beslenir.
% Cikis  : 7 sinif {LFM, NLFM, FSK, Barker, Frank, P_All, none} icin olasilik.
%
% Bu betik, gurultulu veriyle egitilen referans (baz) siniflandiriciyi uretir;
% gurultu giderilmis versiyon icin train_classifier_denoised.m kullanilir.

clear; clc;

%% === Parametreler ===
snr_tag   = "mixed";

classes_raw = {'LFM','NLFM','FSK','Barker','Frank','P_All','none'};
classes     = {'LFM','NLFM','FSK','Barker','Frank','P_All','none'};
rawNumClasses = numel(classes_raw);
numClasses    = numel(classes);

% GoogLeNet 3 kanal bekler; gri yamalar egitimde 1->3 kanala kopyalanir.
inputSize = [224 224 3];

miniBatch = 64;
maxEpochs = 60;
initialLR = 3e-4;
warmupEpochs = 5;
clipGrad = 1.0;

resumeFromCheckpoint = true;
resumeFromModel      = true;
useGPU  = canUseGPU;

checkpointPeriod = 1;

%% === Dataset yolları (gri) ===
% Tum yollar bu betigin konumundan (repo koku) turetilir; makineye ozel
% mutlak yol yoktur.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
dataRoot    = fullfile(projectRoot, 'dataset', 'dataset_multiframe_patches');
specRoot    = fullfile(dataRoot, 'specNoisyGray');            % gurultulu gri yamalar
labelsPath  = fullfile(dataRoot, 'labels.csv');

% Cikti klasorleri proje kokunde tutulur.
modelsDir   = fullfile(projectRoot, 'models');
resultsDir  = fullfile(projectRoot, 'results');
ckptDir     = fullfile(projectRoot, 'checkpoints');
if ~exist(resultsDir,'dir'); mkdir(resultsDir); end
if ~exist(modelsDir,'dir');  mkdir(modelsDir);  end
if ~exist(ckptDir,'dir');    mkdir(ckptDir);    end

modelFile      = fullfile(modelsDir, 'classifier.mat');
checkpointFile = fullfile(ckptDir,   'ckpt_classifier.mat');

fprintf('\n==== Gri spektrogram siniflandirici egitimi (GoogLeNet) ====\n');
fprintf('Cihaz: %s\n', ternary(useGPU,'GPU','CPU'));

%% === labels.csv'yi yükle ===
labelTbl  = readtable(labelsPath);
fileVar   = labelTbl.Properties.VariableNames{1};
filenames = string(labelTbl.(fileVar));

Yraw = single(table2array(labelTbl(:,2:end-3)));  % N x 7 sinif etiketi (son 3 sutun meta veri)

if size(Yraw,2) ~= rawNumClasses
    error('labels.csv class kolon sayisi (%d) ile classes_raw (%d) uyumsuz!', ...
        size(Yraw,2), rawNumClasses);
end

Yfull = Yraw;
if size(Yfull,2) ~= numClasses
    error('Label kolon sayisi (%d) ile classes (%d) uyumsuz!', ...
        size(Yfull,2), numClasses);
end

N = numel(filenames);

%% === PNG yollarını bul (artık direkt specGray altında) ===
imagePaths = strings(N,1);
for i = 1:N
    fname = filenames(i);
    [~,~,ext] = fileparts(fname);
    if strlength(ext) == 0
        fname = fname + ".png";
    end

    p = fullfile(specRoot, fname);
    if exist(p, "file")
        imagePaths(i) = p;
    else
        % fallback: alt klasör varsa diye
        info = dir(fullfile(specRoot, "**", fname));
        if isempty(info)
            error('PNG bulunamadi: %s (specRoot altinda arandi)', fname);
        end
        imagePaths(i) = fullfile(info(1).folder, info(1).name);
    end
end

%% === Datastore ===
imds = imageDatastore(imagePaths, ...
    'ReadFcn', @(x) imread(x));  % GRAY (HxW uint8) gelecek

ds = combine(imds, arrayDatastore(Yfull));

%% === Train/Val böl ===
idx = randperm(N);
Ntrain = floor(0.8 * N);
trainIdx = idx(1:Ntrain);
valIdx   = idx(Ntrain+1:end);

dsTrain = subset(ds, trainIdx);
dsVal   = subset(ds, valIdx);

%% === Class weights ===
posRate = max(sum(Yfull(trainIdx,:),1) ./ numel(trainIdx), 1e-3);
wPos = 0.5 ./ posRate;
wNeg = 0.5 ./ (1 - posRate);
classWeights = single([wPos; wNeg]);   % 2 x C

%% === Model & optimizer init / RESUME ===
iteration = 0;
Tloss = [];
Vloss = [];
bestVal = +inf;
bestNet = [];
startEpoch = 1;

if resumeFromCheckpoint && exist(checkpointFile,"file")
    S = load(checkpointFile);
    fprintf('Checkpoint yüklendi: %s\n', checkpointFile);

    dlnet       = S.dlnet;
    if isfield(S,'trailingAvg'); trailingAvg = S.trailingAvg; else; trailingAvg = []; end
    if isfield(S,'trailingVar'); trailingVar = S.trailingVar; else; trailingVar = []; end
    if isfield(S,'iteration');   iteration   = S.iteration;   end
    if isfield(S,'Tloss');       Tloss       = S.Tloss;       end
    if isfield(S,'Vloss');       Vloss       = S.Vloss;       end
    if isfield(S,'bestVal');     bestVal     = S.bestVal;     end
    if isfield(S,'bestNet');     bestNet     = S.bestNet;     else; bestNet = dlnet; end
    if isfield(S,'epoch');       startEpoch  = S.epoch + 1;   else; startEpoch = 1;  end

elseif resumeFromModel && exist(modelFile,"file")
    S = load(modelFile);
    if isfield(S,'bestNet')
        fprintf('Model dosyasindan (bestNet) baslatiliyor: %s\n', modelFile);
        dlnet = S.bestNet;
    else
        fprintf('Model dosyasinda bestNet yok, sifirdan baslanacak.\n');
        lgraph = localGoogLeNet(inputSize, numClasses);
        dlnet  = dlnetwork(lgraph);
    end
    trailingAvg = [];
    trailingVar = [];
    bestNet = dlnet;
    startEpoch = 1;

else
    fprintf('Checkpoint/model bulunamadi, sifirdan egitim.\n');
    lgraph = localGoogLeNet(inputSize, numClasses);
    dlnet  = dlnetwork(lgraph);
    trailingAvg = [];
    trailingVar = [];
    bestNet = dlnet;
    startEpoch = 1;
end

if useGPU
    dlnet = dlupdate(@gpuArray, dlnet);
    if ~isempty(trailingAvg), trailingAvg = dlupdate(@gpuArray, trailingAvg); end
    if ~isempty(trailingVar), trailingVar = dlupdate(@gpuArray, trailingVar); end
end

%% === minibatchqueue ===
mbqTrain = minibatchqueue(dsTrain, ...
    "MiniBatchSize", miniBatch, ...
    "MiniBatchFcn", @(x,y) toDLWithAug_GRAY(x,y,inputSize,true,useGPU), ...
    "MiniBatchFormat", ["SSCB","CB"], ...
    "PartialMiniBatch", "discard", ...
    "OutputCast", "single", ...
    "OutputEnvironment","auto");

mbqVal = minibatchqueue(dsVal, ...
    "MiniBatchSize", miniBatch, ...
    "MiniBatchFcn", @(x,y) toDLWithAug_GRAY(x,y,inputSize,false,useGPU), ...
    "MiniBatchFormat", ["SSCB","CB"], ...
    "PartialMiniBatch", "discard", ...
    "OutputCast", "single", ...
    "OutputEnvironment","auto");

%% === Eğitim döngüsü ===
for epoch = startEpoch:maxEpochs

    if epoch <= warmupEpochs
        learnRate = initialLR * epoch / max(1,warmupEpochs);
    else
        progress = (epoch - warmupEpochs) / (maxEpochs - warmupEpochs);
        learnRate = 0.5*initialLR*(1+cos(pi*progress));
    end

    reset(mbqTrain);
    lossEpoch = 0; count=0;

    while hasdata(mbqTrain)
        iteration = iteration + 1;
        [X, T] = next(mbqTrain);

        [loss, gradients] = dlfeval(@localModelLoss, dlnet, X, T, classWeights);

        gradients = dlupdate(@(g) boundGrad(g, clipGrad), gradients);

        [dlnet, trailingAvg, trailingVar] = adamupdate(dlnet, gradients, ...
            trailingAvg, trailingVar, iteration, learnRate, 0.9, 0.999, 1e-8);

        lossEpoch = lossEpoch + double(gather(extractdata(loss)));
        count = count + 1;
    end
    trainLoss = lossEpoch / max(count,1);
    Tloss(end+1) = trainLoss;

    valLoss = 0; vcount=0;
    reset(mbqVal);
    while hasdata(mbqVal)
        [Xv, Tv] = next(mbqVal);
        Yv = forward(dlnet, Xv, "Outputs","sigmoid");
        Yv = toCB(Yv);
        lv = localBCELoss(Yv, Tv, classWeights);
        valLoss = valLoss + double(gather(extractdata(lv)));
        vcount = vcount + 1;
    end
    valLoss = valLoss / max(vcount,1);
    Vloss(end+1) = valLoss;

    if valLoss < bestVal
        bestVal = valLoss;
        bestNet = dlnet;
        save(modelFile, 'bestNet','classes');
    end

    if mod(epoch, checkpointPeriod) == 0
        save(checkpointFile, ...
            'dlnet','trailingAvg','trailingVar', ...
            'iteration','epoch','Tloss','Vloss', ...
            'bestVal','bestNet','classes','snr_tag');
    end

    fprintf('Epoch %3d/%3d | LR %.2e | TrainLoss %.4f | ValLoss %.4f\n', ...
        epoch, maxEpochs, learnRate, trainLoss, valLoss);
end

%% === Eşik optimizasyonu ve metrikler (Val seti) ===
reset(mbqVal);
YP = []; YT = [];
while hasdata(mbqVal)
    [Xv, Tv] = next(mbqVal);
    Yv = forward(bestNet, Xv, "Outputs","sigmoid");
    Yv = toCB(Yv);
    YP = [YP; gather(extractdata(Yv))']; %#ok<AGROW>
    YT = [YT; gather(extractdata(Tv))']; %#ok<AGROW>
end

optThr = zeros(1,numClasses);
results_all = {};
for c=1:numClasses
    [thr, a,p,r,f] = localOptimizeThreshold(YT(:,c), YP(:,c));
    optThr(c)=thr;
    results_all = [results_all; {char(snr_tag), classes{c}, round(100*a,2), round(p,3), round(r,3), round(f,3)}]; %#ok<AGROW>
end

save(modelFile,'bestNet','classes','optThr','Tloss','Vloss','-append');

results_table = cell2table(results_all, ...
    'VariableNames', {'SNR_tag','Class','Accuracy_percent','Precision','Recall','F1_score'});
resultsXlsx = fullfile(resultsDir, 'results_googlenet_multilabel_gray.xlsx');
writetable(results_table, resultsXlsx);
fprintf('Egitim tamamlandi. Sonuclar: %s\n', resultsXlsx);
end

%% ========================= Yardımcılar =========================

function [loss, grads] = localModelLoss(dlnet, X, T, classWeights)
Y = forward(dlnet, X, "Outputs","sigmoid");
Y = toCB(Y);
loss = localBCELoss(Y, T, classWeights);
grads = dlgradient(loss, dlnet.Learnables);
end

function loss = localBCELoss(Y, T, classWeights)
% Y, T: CxB (sigmoid sonrası)
% classWeights: 2xC  (row1: posW, row2: negW)
%
% loss = BCE_weighted + lambdaJ * (1-mean(IoU)) + lambdaS * (1-mean(softExact))
%
% Notlar:
% - IoU (soft jaccard) false-positive (ekstra sınıf) cezalarını güçlendirir
% - SoftExact (subset-soft) "set tamamen doğru olsun" baskısı ekler

epsi = 1e-6;
Y = min(max(Y, epsi), 1 - epsi);   % clamp
B = size(Y,2);

% =========================
% (1) Agirlikli BCE
% =========================
posW = classWeights(1,:)';   % Cx1
negW = classWeights(2,:)';   % Cx1

posTerm = -posW .* (T .* log(Y));          % CxB
negTerm = -negW .* ((1-T) .* log(1-Y));    % CxB
lossBCE_perClass = sum(posTerm + negTerm, 2) / max(B,1);   % Cx1
lossBCE = mean(lossBCE_perClass);                           % scalar

% =========================
% (2) Soft Jaccard (IoU) set loss
% =========================
% IoU_b = sum(p*y)/sum(p+y-p*y)
inter = sum(Y .* T, 1);                         % 1xB
uni   = sum(Y + T - Y .* T, 1);                 % 1xB
iou   = (inter + epsi) ./ (uni + epsi);         % 1xB
lossJ = 1 - mean(iou);                          % scalar

% =========================
% (3) Soft Subset (Exact-match) set loss
% =========================
% qb = Π_c ( y*p + (1-y)*(1-p) )
Q = T .* Y + (1 - T) .* (1 - Y);                % CxB
logQ = log(Q + epsi);                           % CxB
log_qb = sum(logQ, 1);                          % 1xB
qb = exp(log_qb);                               % 1xB
lossS = 1 - mean(qb);                           % scalar

% =========================
% (4) Birleştir
% =========================
% Ağırlıklar (başlangıç önerisi)
lambdaJ = 0.25;     % 0.1..0.4 arası dene
lambdaS = 0.10;     % 0.05..0.2 arası dene (çok büyütürsen öğrenme zorlaşır)

loss = lossBCE + lambdaJ * lossJ + lambdaS * lossS;
end


function g = boundGrad(g, clip)
if ~isempty(g)
    g = dlupdate(@(x) max(min(x,clip), -clip), g);
end
end

function [thr, acc, prec, rec, f1] = localOptimizeThreshold(yTrue, yProb)
yTrue = logical(yTrue(:));
yProb = yProb(:);
ths = linspace(0.1, 0.9, 33);
bestF = -inf; thr = 0.5; acc=0; prec=0; rec=0; f1=0;
for t = ths
    yp = yProb > t;
    a = mean(yp==yTrue);
    p = sum(yp & yTrue) / (sum(yp) + eps);
    r = sum(yp & yTrue) / (sum(yTrue) + eps);
    f = 2*p*r/(p+r+eps);
    if f>bestF
        bestF=f; thr=t; acc=a; prec=p; rec=r; f1=f;
    end
end
end

function Ycb = toCB(Y)
sz = size(Y);
if numel(sz) == 4
    Ycb = reshape(Y, [sz(3), sz(4)]);
else
    Ycb = Y;
end
end

function [Xdl, Tdl] = toDLWithAug_GRAY(xBatch, yBatch, inputSize, doAug, useGPU)
% GRAY dataset okur (HxW uint8, tek kanal). GoogLeNet 3 kanal ister:
% -> 1 kanalı 3'e kopyalar (repmat)

if iscell(yBatch)
    yMat = cell2mat(cellfun(@(r) reshape(single(r),1,[]), yBatch, 'UniformOutput', false));
else
    yMat = single(yBatch);
end
B = size(yMat,1);

H = inputSize(1); W = inputSize(2); Ch = inputSize(3); %#ok<NASGU>
X = zeros(H, W, 3, B, 'single');

for i = 1:B
    xi = xBatch{i};              % HxW uint8 (tek kanal)
    if ndims(xi) == 3
        xi = xi(:,:,1);          % güvenlik: yanlışlıkla 3 kanal gelirse
    end

    xi = im2single(xi);
    xi = imresize(xi, [H W]);

    % Augment (GRAY üzerinde)
    if doAug
        Wimg = size(xi,2);
        xi = circshift(xi,[0, randi(round([-0.1 0.1]*Wimg))]);
        Himg = size(xi,1);
        xi = circshift(xi,[randi(round([-0.1 0.1]*Himg)), 0]);

        if rand < 0.8
            mW = randi([max(1,round(0.02*Wimg)) max(1,round(0.08*Wimg))]);
            sW = randi([1 max(1,Wimg-mW)]);
            xi(:, sW:sW+mW-1) = min(xi(:));
        end
        if rand < 0.8
            mH = randi([max(1,round(0.05*Himg)) max(1,round(0.2*Himg))]);
            sH = randi([1 max(1,Himg-mH)]);
            xi(sH:sH+mH-1, :) = min(xi(:));
        end
        if rand < 0.15
            xi = imgaussfilt(xi, 0.6);
        end
    end

    % 1 -> 3 kanal kopya (grayscale bilgi)
    xi3 = repmat(xi,1,1,3);

    % Normalizasyon
    xi3 = (xi3 - 0.5) / 0.5;

    X(:,:,:,i) = xi3;
end

if useGPU
    X    = gpuArray(X);
    yMat = gpuArray(yMat);
end

Xdl = dlarray(X, 'SSCB');
Tdl = dlarray(single(yMat'), 'CB');
end

function out = ternary(cond,a,b)
if cond, out = a; else, out = b; end
end

function lgraph = localGoogLeNet(inputSize, numClasses)
net = googlenet;
lgraph = layerGraph(net);

% Önemli: Biz zaten normalize ediyoruz -> input Normalization 'none' olsun
newInput = imageInputLayer(inputSize, ...
    'Name','data', ...
    'Normalization','none');
lgraph = replaceLayer(lgraph,'data', newInput);

layersToRemove = {'loss3-classifier','prob','output'};
lgraph = removeLayers(lgraph, layersToRemove);

newHead = [
    fullyConnectedLayer(numClasses, ...
        'Name','fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10)
    sigmoidLayer('Name','sigmoid')
];
lgraph = addLayers(lgraph, newHead);
lgraph = connectLayers(lgraph, 'pool5-7x7_s1', 'fc');
end
