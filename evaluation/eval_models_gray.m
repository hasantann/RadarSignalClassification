function eval_models_gray()
% Denoiser ve siniflandiricilarin SNR'a gore performans olcumu (yalnizca gri giris).
%
% Olculenler:
%   1) Denoiser (denoiser.mat): PSNR / SSIM
%   2) Gurultulu gri giris uzerinde siniflandirici
%   3) Gurultu giderilmis gri giris uzerinde siniflandirici
%
% Girisler:
%   - specNoisyGray    (siniflandirici-1 ve denoiser girisi)
%   - specCleanGray    (denoiser hedefi)
%   - specDenoisedGray (siniflandirici-2; yoksa denoiser ile uretilir)
%   - labels.csv, meta/*.json (SNR bilgisi)
%
% Ciktilar:
%   - Excel: SNR ve sinif bazli metrik tablolari
%   - PNG grafikleri: outDir altinda

clc; close all;

%% ===================== PATHS =====================
% Tum yollar bu betigin konumundan (repo koku) turetilir; makineye ozel
% mutlak yol yoktur.
projectRoot       = fileparts(fileparts(mfilename('fullpath')));
dataRoot          = fullfile(projectRoot, 'dataset', 'dataset_multiframe_patches');

specNoisyGrayDir  = fullfile(dataRoot,'specNoisyGray');
specCleanGrayDir  = fullfile(dataRoot,'specCleanGray');
specDenoisedDir   = fullfile(dataRoot,'specDenoisedGray');
metaDir           = fullfile(dataRoot,'meta');
labelsPath        = fullfile(dataRoot,'labels.csv');

% --- model dosyalari (proje kokundeki models/ klasoru) ---
denModelFile      = fullfile(projectRoot,'models','denoiser.mat');
clfNoisyFile      = fullfile(projectRoot,'models','classifier.mat');
clfDenoisedFile   = fullfile(projectRoot,'models','classifierOnDenoised.mat');

% --- output ---
outDir = fullfile(dataRoot,'_eval_results_gray');
if ~exist(outDir,'dir'); mkdir(outDir); end
outXlsx = fullfile(outDir,'metrics_v10_gray.xlsx');

%% ===================== LOAD LABELS + SNR FROM METADATA =====================
labelTbl = readtable(labelsPath);
fileVar  = labelTbl.Properties.VariableNames{1};
names    = string(labelTbl.(fileVar));
classes  = string(labelTbl.Properties.VariableNames(2:end-3));
Ytrue    = table2array(labelTbl(:,2:end));  % NxC (0/1)

N = numel(names);
C = numel(classes);

snrVals = nan(N,1);
for i=1:N
    jf = fullfile(metaDir, names(i)+".json");
    if ~exist(jf,'file')
        error("Metadata yok: %s", jf);
    end
    s = jsondecode(fileread(jf));
    if isfield(s,'SNR')
        snrVals(i) = s.SNR;
    elseif isfield(s,'SNR_dB')
        snrVals(i) = s.SNR_dB;
    elseif isfield(s,'snr_dB')
        snrVals(i) = s.snr_dB;
    else
        error("Metadata içinde SNR alanı bulunamadı: %s", jf);
    end
end

snrList = unique(snrVals(:)).';
snrList = sort(snrList,'descend');

%% ===================== LOAD MODELS =====================
denNet   = load_net_from_mat(denModelFile);
clfNoisy = load_net_from_mat(clfNoisyFile);
clfDen   = load_net_from_mat(clfDenoisedFile);

useGPU = canUseGPU;

%% ===================== PREALLOC RESULTS =====================
psnr_noisy   = nan(N,1);
ssim_noisy   = nan(N,1);
psnr_deno    = nan(N,1);
ssim_deno    = nan(N,1);

P_noisy = nan(N,C);
P_deno  = nan(N,C);

%% ===================== MAIN LOOP =====================
fprintf("Evaluating %d samples...\n", N);

for i=1:N
    id = names(i);

    % --- GRAY images ---
    noisyGrayPath = fullfile(specNoisyGrayDir, id+".png");
    cleanGrayPath = fullfile(specCleanGrayDir, id+".png");

    if ~exist(noisyGrayPath,'file') || ~exist(cleanGrayPath,'file')
        error("Noisy/Clean gray PNG yok: %s / %s", noisyGrayPath, cleanGrayPath);
    end

    Ig_noisy = im2single(imread(noisyGrayPath));
    Ig_clean = im2single(imread(cleanGrayPath));

    if size(Ig_noisy,3)~=1, Ig_noisy = rgb2gray(Ig_noisy); end
    if size(Ig_clean,3)~=1, Ig_clean = rgb2gray(Ig_clean); end

    % --- denoised gray (from folder if exists, else run denoiser) ---
    denPath = fullfile(specDenoisedDir, id+".png");
    if exist(denPath,'file')
        Ig_den = im2single(imread(denPath));
        if size(Ig_den,3)~=1, Ig_den = rgb2gray(Ig_den); end
    else
        Ig_den = run_denoiser(denNet, Ig_noisy, useGPU);
        imwrite(im2uint8(clamp01(Ig_den)), denPath);
    end

    % --- denoiser metrics (baseline + after) ---
    psnr_noisy(i) = psnr(Ig_noisy, Ig_clean);
    ssim_noisy(i) = ssim(Ig_noisy, Ig_clean);

    psnr_deno(i)  = psnr(Ig_den,   Ig_clean);
    ssim_deno(i)  = ssim(Ig_den,   Ig_clean);

    % =========================
    % CLASSIFIERS (GRAY INPUTS)
    % =========================

    % Classifier-1: noisy gray
    Inoisy3 = repmat(Ig_noisy,1,1,3);   % 224x224x3
    P_noisy(i,:) = run_classifier_probs(clfNoisy, Inoisy3, C);

    % Classifier-2: denoised gray
    Iden3 = repmat(Ig_den,1,1,3);      % 224x224x3
    P_deno(i,:)  = run_classifier_probs(clfDen, Iden3, C);

    if mod(i,200)==0
        fprintf("  %d / %d\n", i, N);
    end
end

%% ===================== THRESHOLD -> PRED LABELS =====================
thr = 0.5;
Yhat_noisy = probs_to_labels(P_noisy, thr);
Yhat_deno  = probs_to_labels(P_deno,  thr);

%% ===================== METRICS: DENOISER =====================
Tden_snr   = summarize_metric_by_snr(snrVals, psnr_noisy, ssim_noisy, psnr_deno, ssim_deno);
Tden_class = summarize_metric_by_class(classes, Ytrue, psnr_noisy, ssim_noisy, psnr_deno, ssim_deno);

%% ===================== METRICS: CLASSIFIER =====================
Tclf_noisy_snr = multilabel_metrics_by_snr(snrVals, Ytrue, Yhat_noisy);
Tclf_deno_snr  = multilabel_metrics_by_snr(snrVals, Ytrue, Yhat_deno);

Tclf_noisy_cls = per_class_metrics(classes, Ytrue, Yhat_noisy);
Tclf_deno_cls  = per_class_metrics(classes, Ytrue, Yhat_deno);

%% ===================== PLOTS (PNG) =====================
plot_denoiser_curves(Tden_snr, outDir);
plot_classifier_accuracy(Tclf_noisy_snr, Tclf_deno_snr, outDir);
plot_perclass_f1(Tclf_noisy_cls, Tclf_deno_cls, outDir);

%% ===================== EXPORT EXCEL =====================
if exist(outXlsx,'file'); delete(outXlsx); end

writetable(Tden_snr,   outXlsx, 'Sheet','Denoiser_bySNR');
writetable(Tden_class, outXlsx, 'Sheet','Denoiser_byClass');

writetable(Tclf_noisy_snr, outXlsx, 'Sheet','ClfNoisy_bySNR');
writetable(Tclf_deno_snr,  outXlsx, 'Sheet','ClfDeno_bySNR');

writetable(Tclf_noisy_cls, outXlsx, 'Sheet','ClfNoisy_byClass');
writetable(Tclf_deno_cls,  outXlsx, 'Sheet','ClfDeno_byClass');

Tsample = table(names, snrVals, psnr_noisy, ssim_noisy, psnr_deno, ssim_deno);
writetable(Tsample, outXlsx, 'Sheet','PerSample_Denoiser');

disp("Degerlendirme tamamlandi.");
disp("Excel: " + outXlsx);
disp("PNG  : " + outDir);

end

%% ========================================================================
% Helpers
% ========================================================================

function net = load_net_from_mat(matFile)
S = load(matFile);
cand = ["net","trainedNet","denNet","clfNet","network","dlnet","bestNet"];
for k=1:numel(cand)
    if isfield(S, cand(k))
        net = S.(cand(k));
        return;
    end
end
fn = fieldnames(S);
if numel(fn)==1
    net = S.(fn{1});
    return;
end
error("Model .mat içinde net bulunamadı: %s", matFile);
end

function Iden = run_denoiser(denNet, Ig_noisy, useGPU)
% Denoiser input: gray -> 3ch (safe)
I3 = repmat(Ig_noisy,1,1,3);

% Try predict first (SeriesNetwork/DAGNetwork)
try
    Y = predict(denNet, I3);
    if ndims(Y)==3 && size(Y,3)>1
        Y = rgb2gray(Y);
    end
    Iden = clamp01(single(Y(:,:,1)));
    return;
catch
end

% dlnetwork fallback
dlX = dlarray(single(I3),'SSCB');
if useGPU, dlX = gpuArray(dlX); end
dlY = forward(denNet, dlX);
Y = gather(extractdata(dlY));

if ndims(Y)==3 && size(Y,3)>1
    Y = rgb2gray(Y);
end
Iden = clamp01(single(Y(:,:,1)));
end

function p = run_classifier_probs(clfNet, I, C)
% Returns 1xC probability vector
dlX = dlarray(single(I),"SSCB");
if canUseGPU, dlX = gpuArray(dlX); end

dlY = forward(clfNet, dlX, "Outputs","sigmoid");
p = gather(extractdata(dlY));

p = squeeze(p);
if isvector(p)
    p = p(:).';
end
% If network returns something like Nx1x1xC
if numel(p) ~= C
    p = reshape(p, 1, []);
end
if numel(p) ~= C
    error("Classifier output size mismatch: got %d, expected %d", numel(p), C);
end
p = double(p);
end

function p = reshape_to_row(p)
if isvector(p)
    p = p(:).';
else
    p = reshape(p, 1, []);
end
p = double(p);
end

function p = fix_prob_vector(p, C)
if numel(p) ~= C
    p = reshape(p, 1, []);
end
if numel(p) ~= C
    error("Classifier output size mismatch: got %d, expected %d", numel(p), C);
end
end

function Yhat = probs_to_labels(P, thr)
Yhat = P >= thr;
z = find(sum(Yhat,2)==0);
for i=1:numel(z)
    [~,k] = max(P(z(i),:));
    Yhat(z(i),k) = 1;
end
end

function T = summarize_metric_by_snr(snrVals, psnrN, ssimN, psnrD, ssimD)
snrList = sort(unique(snrVals),'descend');
rows = [];
for s=1:numel(snrList)
    snr = snrList(s);
    idx = (snrVals==snr);
    rows = [rows; {snr, ...
        mean(psnrN(idx),'omitnan'), std(psnrN(idx),'omitnan'), ...
        mean(ssimN(idx),'omitnan'), std(ssimN(idx),'omitnan'), ...
        mean(psnrD(idx),'omitnan'), std(psnrD(idx),'omitnan'), ...
        mean(ssimD(idx),'omitnan'), std(ssimD(idx),'omitnan')}]; %#ok<AGROW>
end
T = cell2table(rows, 'VariableNames', ...
    {'SNR_dB','PSNR_noisy_mean','PSNR_noisy_std','SSIM_noisy_mean','SSIM_noisy_std',...
             'PSNR_deno_mean','PSNR_deno_std','SSIM_deno_mean','SSIM_deno_std'});
end

function T = summarize_metric_by_class(classes, Ytrue, psnrN, ssimN, psnrD, ssimD)
C = numel(classes);
rows = cell(C,1);
for c=1:C
    idx = (Ytrue(:,c)==1);
    rows{c,1} = {classes(c), ...
        mean(psnrN(idx),'omitnan'), std(psnrN(idx),'omitnan'), ...
        mean(ssimN(idx),'omitnan'), std(ssimN(idx),'omitnan'), ...
        mean(psnrD(idx),'omitnan'), std(psnrD(idx),'omitnan'), ...
        mean(ssimD(idx),'omitnan'), std(ssimD(idx),'omitnan'), ...
        nnz(idx)};
end
rows = vertcat(rows{:});
T = cell2table(rows, 'VariableNames', ...
    {'Class','PSNR_noisy_mean','PSNR_noisy_std','SSIM_noisy_mean','SSIM_noisy_std',...
             'PSNR_deno_mean','PSNR_deno_std','SSIM_deno_mean','SSIM_deno_std','Support'});
end

function Tsnr = multilabel_metrics_by_snr(snrVals, Ytrue, Yhat)
snrList = sort(unique(snrVals),'descend');
rows = [];
for s=1:numel(snrList)
    snr = snrList(s);
    idx = (snrVals==snr);
    [subsetAcc, microP, microR, microF1, macroP, macroR, macroF1] = multilabel_summary(Ytrue(idx,1:end-3), Yhat(idx,:));
    rows = [rows; {snr, subsetAcc, microP, microR, microF1, macroP, macroR, macroF1, nnz(idx)}]; %#ok<AGROW>
end
Tsnr = cell2table(rows, 'VariableNames', ...
    {'SNR_dB','SubsetAccuracy','MicroPrecision','MicroRecall','MicroF1','MacroPrecision','MacroRecall','MacroF1','NumSamples'});
end

function Tcls = per_class_metrics(classes, Ytrue, Yhat)
C = numel(classes);
rows = cell(C,1);
for c=1:C
    yt = Ytrue(:,c)==1;
    yp = Yhat(:,c)==1;

    TP = nnz( yt &  yp);
    FP = nnz(~yt &  yp);
    FN = nnz( yt & ~yp);

    prec = TP / max(TP+FP,1);
    rec  = TP / max(TP+FN,1);
    f1   = 2*prec*rec / max(prec+rec, eps);

    rows{c,1} = {classes(c), TP, FP, FN, prec, rec, f1, nnz(yt)};
end
rows = vertcat(rows{:});
Tcls = cell2table(rows, 'VariableNames', {'Class','TP','FP','FN','Precision','Recall','F1','Support'});
end

function [subsetAcc, microP, microR, microF1, macroP, macroR, macroF1] = multilabel_summary(Yt, Yp)
subsetAcc = mean(all(Yt==Yp,2));

C = size(Yt,2);
precC = zeros(C,1);
recC  = zeros(C,1);
f1C   = zeros(C,1);

TPs=0; FPs=0; FNs=0;

for c=1:C
    yt = Yt(:,c)==1; yp = Yp(:,c)==1;
    TP = nnz( yt &  yp);
    FP = nnz(~yt &  yp);
    FN = nnz( yt & ~yp);

    TPs = TPs + TP;
    FPs = FPs + FP;
    FNs = FNs + FN;

    precC(c) = TP / max(TP+FP,1);
    recC(c)  = TP / max(TP+FN,1);
    f1C(c)   = 2*precC(c)*recC(c) / max(precC(c)+recC(c), eps);
end

microP  = TPs / max(TPs+FPs,1);
microR  = TPs / max(TPs+FNs,1);
microF1 = 2*microP*microR / max(microP+microR, eps);

macroP  = mean(precC);
macroR  = mean(recC);
macroF1 = mean(f1C);
end

function plot_denoiser_curves(Tden_snr, outDir)
snr = Tden_snr.SNR_dB;

figure('Name','PSNR vs SNR');
plot(snr, Tden_snr.PSNR_noisy_mean,'-o'); hold on;
plot(snr, Tden_snr.PSNR_deno_mean ,'-o'); grid on;
xlabel('SNR (dB)'); ylabel('PSNR (dB)');
legend('Noisy->Clean','Denoised->Clean','Location','best');
exportgraphics(gcf, fullfile(outDir,'denoiser_psnr_vs_snr.png'), 'Resolution',200);

figure('Name','SSIM vs SNR');
plot(snr, Tden_snr.SSIM_noisy_mean,'-o'); hold on;
plot(snr, Tden_snr.SSIM_deno_mean ,'-o'); grid on;
xlabel('SNR (dB)'); ylabel('SSIM');
legend('Noisy->Clean','Denoised->Clean','Location','best');
exportgraphics(gcf, fullfile(outDir,'denoiser_ssim_vs_snr.png'), 'Resolution',200);
end

function plot_classifier_accuracy(Tnoisy, Tden, outDir)
snr = Tnoisy.SNR_dB;

figure('Name','SubsetAccuracy vs SNR');
plot(snr, Tnoisy.SubsetAccuracy,'-o'); hold on;
plot(snr, Tden.SubsetAccuracy  ,'-o'); grid on;
xlabel('SNR (dB)'); ylabel('Subset Accuracy (Exact Match)');
legend('Classifier on Noisy Gray','Classifier on Denoised Gray','Location','best');
exportgraphics(gcf, fullfile(outDir,'classifier_subsetacc_vs_snr.png'), 'Resolution',200);

figure('Name','MacroF1 vs SNR');
plot(snr, Tnoisy.MacroF1,'-o'); hold on;
plot(snr, Tden.MacroF1  ,'-o'); grid on;
xlabel('SNR (dB)'); ylabel('Macro F1');
legend('Classifier on Noisy Gray','Classifier on Denoised Gray','Location','best');
exportgraphics(gcf, fullfile(outDir,'classifier_macrof1_vs_snr.png'), 'Resolution',200);
end

function plot_perclass_f1(Tcls_noisy, Tcls_den, outDir)
[~,ia,ib] = intersect(Tcls_noisy.Class, Tcls_den.Class, 'stable');
A = Tcls_noisy(ia,:);
B = Tcls_den(ib,:);

figure('Name','Per-class F1 Comparison');
bar([A.F1, B.F1]);
grid on;
set(gca,'XTickLabel', cellstr(A.Class), 'XTickLabelRotation',45);
ylabel('F1');
legend('Noisy Gray','Denoised Gray','Location','best');
exportgraphics(gcf, fullfile(outDir,'classifier_perclass_f1.png'), 'Resolution',200);
end

function x = clamp01(x)
x = min(max(x,0),1);
end
