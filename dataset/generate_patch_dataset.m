function generate_patch_dataset()
% Cok cerceveli (multiframe) radar spektrogram yama veri seti uretimi.
%
% Akis: 10 kisa cerceveden olusan sahne uret -> uzun spektrogram (gurultulu +
% temiz) hesapla -> ortusen yamalara bol -> her yama icin PNG, cok etiketli
% etiket ve meta veri kaydet.
%
% Ciktilar:
%   outRoot/specNoisyGray/sample_000001.png   (224x224 uint8, gurultulu)
%   outRoot/specCleanGray/sample_000001.png   (224x224 uint8, temiz)
%   outRoot/meta/sample_000001.json
%   outRoot/labels.csv   (ilk sutun dosya adi, ardindan sinif ve meta sutunlari)

clc; close all;
rng(2);

cfg = get_dataset_config();
cfg.datasetV6_multiframe = struct();

% ===================== AYARLAR =====================
opts = struct();
opts.outRoot         = fullfile(cfg.outRoot, "dataset_multiframe_patches"); % yeni dataset kökü
opts.snrList         = cfg.snrList;             % [20 10 0 -10] gibi
opts.numScenesPerSNR = 20;                     % SNR başına sahne sayısı
opts.numFrames       = 10;                      % multiframe = 10x shortframe
opts.Wpatch          = cfg.spec.specW;          % 224
opts.overlap         = 0.50;                    % 0.5 => hop=112
opts.verboseEvery    = 25;
opts.keepNoneProb    = 0.30;                    % none patch'lerin sadece %30'unu kaydet

% Patch hop
opts.hop = round(opts.Wpatch * (1 - opts.overlap));   % 224*(1-0.5)=112

% ===================== KLASÖRLER =====================
specNoisyGrayDir      = fullfile(opts.outRoot, "specNoisyGray");       % noisy
specCleanGrayDir = fullfile(opts.outRoot, "specCleanGray");  % clean
metadir          = fullfile(opts.outRoot, "meta");

if ~exist(specNoisyGrayDir,'dir'),      mkdir(specNoisyGrayDir);      end
if ~exist(specCleanGrayDir,'dir'), mkdir(specCleanGrayDir); end
if ~exist(metadir,'dir'),          mkdir(metadir);          end

% ===================== LABEL TABLO HAZIRLA =====================
classList = cfg.classList;               % {'LFM','NLFM','FSK','Barker','Frank','P_All','none'}
C = numel(classList);

rows = {};   % dinamik biriktirip sonra table yapacağız
rowCount = 0;

globalSampleIdx = 0;

fprintf("\n=== MULTIFRAME PATCH DATASET GENERATION (NOISY + CLEAN) ===\n");
fprintf("OutRoot: %s\n", opts.outRoot);
fprintf("Scenes/SNR: %d | SNRs: %s | overlap=%.2f hop=%d | keepNoneProb=%.2f\n", ...
    opts.numScenesPerSNR, mat2str(opts.snrList), opts.overlap, opts.hop, opts.keepNoneProb);

for sIdx = 1:numel(opts.snrList)
    snr_dB = opts.snrList(sIdx);

    for sceneId = 1:opts.numScenesPerSNR
        if mod(sceneId, opts.verboseEvery)==0
            fprintf("SNR=%+d dB | scene %d/%d\n", snr_dB, sceneId, opts.numScenesPerSNR);
        end

        % ---- 1) Multiframe sahne üret ----
        [iqNoisy, iqClean, sceneMeta] = simulate_scene_multiframe_v6(cfg, snr_dB, opts.numFrames);

        % ---- 2) Long spectrogram gray01 üret (NOISY + CLEAN) + resized eksenler ----
        Wlong = opts.numFrames * cfg.spec.specW;   % 2240
        [ID_noisy_long_gray01, ID_clean_long_gray01, tLong, fLong] = ...
            iq_to_spec_gray01_long_with_axes(iqNoisy, iqClean, cfg, Wlong);

        % ---- 3) Patch'lere böl (NOISY + CLEAN) ----
        [patchesNoisyGray01, starts] = slice_overlapped_patches(ID_noisy_long_gray01, opts.Wpatch, opts.hop);
        patchesCleanGray01           = slice_overlapped_patches(ID_clean_long_gray01, opts.Wpatch, opts.hop);

        % ---- 4) Her patch için label + metadata + PNG kaydet ----
        for pIdx = 1:numel(patchesNoisyGray01)

            sCol = starts(pIdx);
            eCol = sCol + opts.Wpatch - 1;

            % patch time aralığı (saniye)
            tStart = tLong(sCol);
            tStop  = tLong(eCol);

            % ---- patch-level multi-label çıkar ----
            y = labels_for_patch_from_scene(sceneMeta, classList, tStart, tStop);

            % ---- none kuralı ----
            if all(y(1:end-1)==0)
                y(end) = 1; % none
            else
                y(end) = 0;
            end

            % ---- keepNoneProb filtresi (noise-only patlamasını engeller) ----
            isNonePatch = (sum(y(1:end-1)) == 0);
            if isNonePatch && (rand > opts.keepNoneProb)
                continue; % bu patch'i kaydetme (noisy/clean/png/json/csv yok)
            end

            % ---- sample id ----
            globalSampleIdx = globalSampleIdx + 1;
            fname = sprintf("sample_%06d", globalSampleIdx);

            % ---- PNG kaydet (NOISY) ----
            patchNoisy_u8 = im2uint8(patchesNoisyGray01{pIdx});   % [0,1] -> uint8
            imwrite(patchNoisy_u8, fullfile(specNoisyGrayDir, fname + ".png"));

            % ---- PNG kaydet (CLEAN) ----
            patchClean_u8 = im2uint8(patchesCleanGray01{pIdx});   % [0,1] -> uint8
            imwrite(patchClean_u8, fullfile(specCleanGrayDir, fname + ".png"));

            % ---- metadata.json ----
            patchMeta = struct();
            patchMeta.filename      = char(fname + ".png");
            patchMeta.sample_id     = globalSampleIdx;
            patchMeta.scene_id      = sceneId;
            patchMeta.snr_dB        = snr_dB;

            patchMeta.patch_index   = pIdx;
            patchMeta.start_col     = sCol;
            patchMeta.end_col       = eCol;

            patchMeta.t_start_s     = tStart;
            patchMeta.t_stop_s      = tStop;

            patchMeta.f_start_Hz    = fLong(1);
            patchMeta.f_stop_Hz     = fLong(end);

            patchMeta.classList     = classList;
            patchMeta.labels        = y;

            % scene detayını da koy (debug/analiz için çok işe yarar)
            patchMeta.scene         = sceneMeta;

            js  = jsonencode(patchMeta, "PrettyPrint", true);
            fid = fopen(fullfile(metadir, fname + ".json"), "w");
            fwrite(fid, js, "char");
            fclose(fid);

            % ---- labels.csv satırı biriktir ----
            rowCount = rowCount + 1;
            oneRow = cell(1, 1 + C + 3);
            oneRow{1} = char(fname);            % uzantısız
            for cc = 1:C
                oneRow{1+cc} = y(cc);
            end
            oneRow{1+C+1} = snr_dB;
            oneRow{1+C+2} = sceneId;
            oneRow{1+C+3} = pIdx;

            rows(rowCount,:) = oneRow; %#ok<AGROW>
        end
    end
end

% ===================== labels.csv yaz =====================
varNames = ["filename", string(classList), "SNR_dB","scene_id","patch_id"];
T = cell2table(rows, "VariableNames", cellstr(varNames));
writetable(T, fullfile(opts.outRoot, "labels.csv"));

fprintf("\nDONE. Total patches saved: %d\n", globalSampleIdx);
fprintf("specGray      : %s\n", specNoisyGrayDir);
fprintf("specCleanGray : %s\n", specCleanGrayDir);
fprintf("labels.csv    : %s\n", fullfile(opts.outRoot, "labels.csv"));
end

%% ========================================================================
%  LONG SPECTROGRAM + AXES (resize sonrası time/freq ekseni)  [NOISY + CLEAN]
% ========================================================================
function [ID_noisy_long_gray01, ID_clean_long_gray01, tLong, fLong] = iq_to_spec_gray01_long_with_axes(iq, iqClean, cfg, Wtarget)

[S_noisy_dB, S_clean_dB, tVec, fVec] = compute_spectrogram_fixed(iq, iqClean, cfg.spec);

maxRef = cfg.spec.maxRef_dB;
minRef = maxRef - cfg.spec.dynRange_dB;
den    = maxRef - minRef;

% --- noisy normalize (AYNI global scale) ---
S_norm = (S_noisy_dB - minRef) / max(den, 1e-6);
S_norm = max(min(S_norm,1),0);
ID_noisy_long_gray01 = imresize(S_norm, [cfg.spec.specH, Wtarget], "bilinear");

% --- clean normalize (AYNI global scale) ---
S_norm = (S_clean_dB - minRef) / max(den, 1e-6);
S_norm = max(min(S_norm,1),0);
ID_clean_long_gray01 = imresize(S_norm, [cfg.spec.specH, Wtarget], "bilinear");

% resize sonrası eksenleri lineer map et
tLong = linspace(tVec(1), tVec(end), Wtarget);
fLong = linspace(fVec(1), fVec(end), cfg.spec.specH);

tLong = tLong(:).';  % row
fLong = fLong(:);    % col
end

%% ========================================================================
%  PATCH LABEL: ToA/PW overlap ile patch-level etiket
% ========================================================================
function y = labels_for_patch_from_scene(sceneMeta, classList, tStart, tStop)
C = numel(classList);
y = zeros(1,C);

if ~isfield(sceneMeta,'signals') || isempty(sceneMeta.signals)
    return;
end

signals = sceneMeta.signals;

for k = 1:numel(signals)
    cls = string(signals(k).class);

    idx = find(strcmp(classList, cls), 1);
    if isempty(idx), continue; end
    if strcmp(classList{idx}, 'none'), continue; end

    ToA = signals(k).ToA_list(:);
    PW  = signals(k).parameters.PW;

    pulseStart = ToA;
    pulseEnd   = ToA + PW;

    hit = any( (pulseStart < tStop) & (pulseEnd > tStart) );
    if hit
        y(idx) = 1;
    end
end
end

%% ========================================================================
%  MULTI-FRAME SCENE SIMULATION (10x shortframe)
% ========================================================================
function [iqNoisy, iqCleanTotal, frameInfo] = simulate_scene_multiframe_v6(cfg, snr_dB, numFrames)

Nshort = cfg.N;
N      = numFrames * Nshort;
fs     = cfg.spec.fs;
t      = (0:N-1).' / fs;

frameDurShort = cfg.frameDur;
frameDur      = numFrames * frameDurShort;

iqCleanTotal = complex(zeros(N,1));

% Empty frame?
isEmpty = (rand < cfg.pEmptyFrame);

typePool = {'LFM','NLFM','FSK','Barker','Frank','P_All'};

if isEmpty
    selectedTypes = {};
else
    numTypes = randi([1, cfg.maxClassesPerFrame]);
    perm = randperm(numel(typePool), numTypes);
    selectedTypes = typePool(perm);
end

% Fc guard-band
usedFc     = [];
minDeltaFc = cfg.minDeltaFc;
maxTrials  = 50;

signalList = struct([]);
sigCount   = 0;

for k = 1:numel(selectedTypes)
    className = selectedTypes{k};

    params = generate_parameters_for_class_multiframe(cfg, className, frameDur);

    % Fc select (guard band)
    ok = false; trial = 0;
    while ~ok && trial < maxTrials
        trial = trial + 1;
        params.Fc = rand_range(cfg.Fc_range);

        if isempty(usedFc)
            ok = true;
        else
            dFc = abs(params.Fc - usedFc);
            ok  = all(dFc >= minDeltaFc);
        end
    end
    if ~ok
        warning('Guard band saglanamadi, son Fc kullanildi.');
    end
    usedFc(end+1,1) = params.Fc; %#ok<AGROW>

    % Pulse train
    [xClean, ToA_list] = simulate_pulsetrain_by_class_v6(cfg, className, params, t, frameDur);
    iqCleanTotal = iqCleanTotal + xClean;

    % Meta
    sigCount = sigCount + 1;
    signalList(sigCount).id         = sigCount;
    signalList(sigCount).class      = className;
    signalList(sigCount).subtype    = params.subtype;
    signalList(sigCount).parameters = params;
    signalList(sigCount).ToA_list   = ToA_list(:).';
end

% Noise
refPower   = cfg.sigRefPower;
noisePower = refPower / (10^(snr_dB/10));
noise      = sqrt(noisePower/2) * (randn(N,1) + 1j*randn(N,1));
iqNoisy    = iqCleanTotal + noise;

frameInfo = struct();
frameInfo.SNR_dB     = snr_dB;
frameInfo.fs         = fs;
frameInfo.N          = N;
frameInfo.frameDur   = frameDur;
frameInfo.numSignals = sigCount;
frameInfo.signals    = signalList;
end

%% ========================================================================
%  PARAM / PULSE GENERATORS
% ========================================================================
function params = generate_parameters_for_class_multiframe(cfg, className, frameDur)
params = struct();
params.PW  = rand_range(cfg.PW_range);

priMin = cfg.PRI_range(1);
priMax = cfg.PRI_range(2);

priMax2 = (frameDur - params.PW)/2;
if priMax2 > priMin
    priMaxUse = min(priMax, priMax2);
else
    priMaxUse = priMax;
end
params.PRI = rand_range([priMin, priMaxUse]);

params.mode       = "pulsetrain";
params.pulseShape = "rect";
params.ToA0 = rand * min(params.PRI, max(0, frameDur - params.PW));

lastStart = frameDur - params.PW;
if lastStart <= 0
    params.numPulses = 0;
else
    params.numPulses = floor((lastStart - params.ToA0)/params.PRI) + 1;
    params.numPulses = max(params.numPulses, 1);
end

params.subtype = string(className);

switch className
    case 'LFM'
        params.BW = rand_range(cfg.BW_chirp_range);
        if rand < 0.5, params.slope = "Up"; else, params.slope = "Down"; end
    case 'NLFM'
        params.BW    = rand_range(cfg.BW_chirp_range);
        params.shape = "Cubic";
    case 'FSK'
        if rand < 0.5
            params.subtype   = "Costas";
            params.fskMode   = "Costas";
            params.M         = cfg.Costas_M_list(randi(numel(cfg.Costas_M_list)));
            params.tone_spacing = rand_range(cfg.FSK_spacing_range);
            params.BW        = params.M * params.tone_spacing;
            params.tone_sequence = randperm(params.M);
        else
            params.subtype = "NonCostas";
            params.M       = randi([5, 10]);
            params.tone_spacing = rand_range(cfg.FSK_spacing_range);
            params.BW      = params.M * params.tone_spacing;
            params.fskMode = "NonCostas_Fixed";
            params.tone_sequence = randperm(params.M, params.M);
        end
    case 'Barker'
        params.pulseShape  = "rect";
        params.code_length = cfg.Barker_lengths(randi(numel(cfg.Barker_lengths)));
        params.chip_time   = params.PW / params.code_length;
        params.BW          = 1 / params.chip_time;
    case 'Frank'
        params.pulseShape = "rect";
        params.Ncode      = cfg.Frank_N_list(randi(numel(cfg.Frank_N_list)));
        total_chips       = params.Ncode^2;
        params.chip_time  = params.PW / total_chips;
        params.BW         = 1 / params.chip_time;
    case 'P_All'
        params.pulseShape = "rect";
        pTypes = ["P1","P2","P3","P4"];
        params.subtype = pTypes(randi(numel(pTypes)));
        if params.subtype == "P1" || params.subtype == "P2"
            params.code_length = randi([32, 64]);
        else
            params.code_length = randi([32, 128]);
        end
        params.chip_time = params.PW / params.code_length;
        params.BW        = 1 / params.chip_time;
    otherwise
        error("Bilinmeyen class: %s", className);
end

params.Fc = 0;
end

function [xTotal, ToA_list] = simulate_pulsetrain_by_class_v6(cfg, className, params, t, frameDur)
xTotal = complex(zeros(size(t)));
if params.numPulses <= 0
    ToA_list = [];
    return;
end
ToA_list = params.ToA0 + (0:params.numPulses-1) * params.PRI;
ToA_list = ToA_list(ToA_list >= 0 & (ToA_list + params.PW) <= frameDur);

for n = 1:numel(ToA_list)
    p2     = params;
    p2.ToA = ToA_list(n);
    p2.pulseIndex = n;
    xTotal = xTotal + simulate_onepulse_by_class(cfg, className, p2, t);
end
end

function x = simulate_onepulse_by_class(cfg, className, params, t)
switch className
    case 'LFM',    x = simulate_lfm_onepulse(params, t, cfg);
    case 'NLFM',   x = simulate_nlfm_onepulse(params, t, cfg);
    case 'FSK'
        if isfield(params,'fskMode') && params.fskMode == "Costas"
            x = simulate_costas_onepulse(params, t, cfg);
        else
            x = simulate_fsk_noncostas_onepulse(params, t, cfg);
        end
    case 'Barker', x = simulate_barker_onepulse(params, t, cfg);
    case 'Frank',  x = simulate_frank_onepulse(params, t, cfg);
    case 'P_All'
        switch params.subtype
            case "P1", x = simulate_p_polyphase_onepulse(params,t,cfg,"P1");
            case "P2", x = simulate_p_polyphase_onepulse(params,t,cfg,"P2");
            case "P3", x = simulate_p_polyphase_onepulse(params,t,cfg,"P3");
            case "P4", x = simulate_p_polyphase_onepulse(params,t,cfg,"P4");
            otherwise, error("Unknown P subtype");
        end
    otherwise
        error("Unknown className: %s", className);
end
end

%% ========================================================================
%  SIGNAL GENERATORS (ONE PULSE)
% ========================================================================
function x = simulate_lfm_onepulse(params, t, cfg)
t = t(:); x = complex(zeros(size(t)));

PW = params.PW; t0 = params.ToA; t1 = t0 + PW;
idx = (t >= t0) & (t < t1);
if ~any(idx), return; end
tau = t(idx) - t0;

if isfield(params,"slope") && params.slope == "Down"
    k = -params.BW / PW;
else
    k =  params.BW / PW;
end

Fc = params.Fc;
phase = 2*pi*( Fc.*tau + 0.5*k.*tau.^2 );
x(idx) = exp(1j*phase);

x = apply_pulse_post(x, idx, cfg);
end

function x = simulate_nlfm_onepulse(params, t, cfg)
t = t(:); x = complex(zeros(size(t)));

PW = params.PW; t0 = params.ToA; t1 = t0 + PW;
idx = (t >= t0) & (t < t1);
if ~any(idx), return; end
tau = t(idx) - t0;
u   = tau / PW;

Fc = params.Fc; BW = params.BW;
if numel(t) < 2, return; end
Ts = t(2) - t(1);

switch params.shape
    case "Cubic"
        u_norm = (u - 0.5) / 0.5;
        f_dev  = (BW/2) * (u_norm.^3);
    otherwise
        f_dev  = (BW/2) * sin(2*pi*(u - 0.5));
end

f_inst = Fc + f_dev;
dphi   = 2*pi * f_inst * Ts;
phase  = cumsum(dphi);

x(idx) = exp(1j*phase);

x = apply_pulse_post(x, idx, cfg);
end

function x = simulate_costas_onepulse(params, t, cfg)
t = t(:); x = complex(zeros(size(t)));

Fc = params.Fc; M = params.M; seq = params.tone_sequence;

t0 = params.ToA; t1 = t0 + params.PW;
idx = (t >= t0) & (t < t1);
if ~any(idx), return; end
tau = t(idx) - t0;

if isfield(params,'pulseIndex')
    ii = mod(params.pulseIndex-1, numel(seq)) + 1;
else
    ii = 1;
end
idxTone = seq(ii);

center = (M + 1)/2;
k      = idxTone - center;
fTone  = Fc + k * params.tone_spacing;

phase  = 2*pi * fTone .* tau;
x(idx) = exp(1j * phase);

x = apply_pulse_post(x, idx, cfg);
end

function x = simulate_fsk_noncostas_onepulse(params, t, cfg)
t = t(:); x = complex(zeros(size(t)));

Fc = params.Fc; M = params.M;

t0 = params.ToA; t1 = t0 + params.PW;
idx = (t >= t0) & (t < t1);
if ~any(idx), return; end
tau = t(idx) - t0;

if isfield(params,'pulseIndex')
    ii = mod(params.pulseIndex-1, numel(params.tone_sequence)) + 1;
else
    ii = 1;
end

kIdx = params.tone_sequence(ii);

center = (M - 1)/2;
k      = kIdx - center;
fTone  = Fc + k * params.tone_spacing;

phase  = 2*pi * fTone .* tau;
x(idx) = exp(1j * phase);

x = apply_pulse_post(x, idx, cfg);
end

function x = simulate_barker_onepulse(params, t, cfg)
t = t(:); x = complex(zeros(size(t)));

Fc = params.Fc;
L  = params.code_length;
Tc = params.chip_time;
PW = params.PW;

switch L
    case 7,  barker_code = [1 1 1 -1 -1 1 -1];
    case 11, barker_code = [1 1 1 -1 -1 -1 1 -1 -1 1 -1];
    case 13, barker_code = [1 1 1 1 1 -1 -1 1 1 -1 1 -1 1];
    otherwise, error("Desteklenmeyen Barker kod uzunlugu: %d", L);
end

t0 = params.ToA; t1 = t0 + PW;
idx = (t >= t0) & (t < t1);
if ~any(idx), return; end
tau = t(idx) - t0;

chip_idx = floor(tau / Tc) + 1;
chip_idx = min(max(chip_idx, 1), L);
chip_vals = barker_code(chip_idx).';

if isfield(cfg,'useChipShaping') && cfg.useChipShaping
    alpha = getf(cfg,'chipTaperAlpha',0.10);
    tau_chip = tau - (chip_idx-1).*Tc;
    wchip = chip_tukey_weight(tau_chip, Tc, alpha);
    chip_vals = chip_vals .* wchip;
end

carrier_phase = 2*pi*Fc .* tau;
x(idx) = chip_vals .* exp(1j*carrier_phase);

x = apply_pulse_post(x, idx, cfg);
end

function x = simulate_frank_onepulse(params, t, cfg)
t = t(:); x = complex(zeros(size(t)));

Fc = params.Fc; N = params.Ncode; PW = params.PW; Tc = params.chip_time;
L = N^2;

[mIdx, nIdx] = ndgrid(0:N-1, 0:N-1);
phiMat = 2*pi .* (mIdx .* nIdx) / N;
phiVec = reshape(phiMat.', 1, []);
codeSeq = exp(1j * phiVec);

t0 = params.ToA; t1 = t0 + PW;
idx = (t >= t0) & (t < t1);
if ~any(idx), return; end
tau = t(idx) - t0;

chip_idx = floor(tau / Tc) + 1;
chip_idx = min(max(chip_idx, 1), L);
baseband = codeSeq(chip_idx).';

if isfield(cfg,'useChipShaping') && cfg.useChipShaping
    alpha = 0.20;
    tau_chip = tau - (chip_idx-1).*Tc;
    wchip = chip_tukey_weight(tau_chip, Tc, alpha);
    baseband = baseband .* wchip;
end

carrier_phase = 2*pi*Fc .* tau;
x(idx) = baseband .* exp(1j*carrier_phase);

x = apply_pulse_post(x, idx, cfg);
end

function x = simulate_p_polyphase_onepulse(params, t, cfg, pType)
t = t(:); x = complex(zeros(size(t)));

Fc = params.Fc; L = params.code_length; Tc = params.chip_time; PW = params.PW;

n = 0:L-1;
switch pType
    case "P1", phi = pi * (n.^2) / L;
    case "P2", phi = pi * (n .* (n - 1)) / L;
    case "P3", phi = pi * (n.^2) / L;
    case "P4", phi = pi * ( n .* (n - L) ) / L;
    otherwise, error("Unknown P type");
end
codeSeq = exp(1j * phi);

t0 = params.ToA; t1 = t0 + PW;
idx = (t >= t0) & (t < t1);
if ~any(idx), return; end
tau = t(idx) - t0;

chip_idx = floor(tau / Tc) + 1;
chip_idx = min(max(chip_idx, 1), L);
baseband = codeSeq(chip_idx).';

if isfield(cfg,'useChipShaping') && cfg.useChipShaping
    alpha = getf(cfg,'chipTaperAlpha',0.10);
    tau_chip = tau - (chip_idx-1).*Tc;
    wchip = chip_tukey_weight(tau_chip, Tc, alpha);
    baseband = baseband .* wchip;
end

carrier_phase = 2*pi*Fc .* tau;
x(idx) = baseband .* exp(1j*carrier_phase);

x = apply_pulse_post(x, idx, cfg);
end

function x = apply_pulse_post(x, idx, cfg)
if isfield(cfg,'useTaper') && cfg.useTaper
    alphaP = getf(cfg,'pulseTaperAlpha',0.10);
    w = tukeywin(nnz(idx), alphaP);
    x(idx) = x(idx) .* w;
end
if isfield(cfg,'usePulsePowerNorm') && cfg.usePulsePowerNorm
    p = mean(abs(x(idx)).^2);
    if p > 0
        x(idx) = x(idx) * sqrt(cfg.pulseTargetPower / p);
    end
end
end

%% ========================================================================
%  PATCH SLICE
% ========================================================================
function [patches, starts] = slice_overlapped_patches(ID_long, W, hop)
T = size(ID_long,2);
starts = 1:hop:(T - W + 1);
patches = cell(1,numel(starts));
for i=1:numel(starts)
    s = starts(i);
    patches{i} = ID_long(:, s:(s+W-1));
end
end

%% ========================================================================
%  SPECTROGRAM FIXED
% ========================================================================
function [S_dB, S_Clean_dB, tVec, fVec] = compute_spectrogram_fixed(iq, iqClean, specCfg)
fs   = specCfg.fs;
nfft = specCfg.nfft;
wlen = specCfg.wlen;
ovlp = specCfg.ovlp;

win = hamming(wlen,'periodic');

[S_noisy, fVec, tVec] = spectrogram(iq,      win, ovlp, nfft, fs, 'centered');
[S_clean, ~,   ~    ] = spectrogram(iqClean, win, ovlp, nfft, fs, 'centered');

S_dB       = 20*log10(abs(S_noisy) + 1e-12);
S_Clean_dB = 20*log10(abs(S_clean) + 1e-12);
end

%% ========================================================================
%  CFG
% ========================================================================
function cfg = get_dataset_config()

% Cikti koku bu betigin konumundan (repo koku) turetilir; mutlak yol yoktur.
cfg.outRoot = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'dataset');

cfg.snrList          = [20 10 0 -10];
cfg.sigRefPower      = 1.0;
cfg.numSamplesPerSNR = 2000;
cfg.usePulsePowerNorm = true;
cfg.pulseTargetPower = 1.0;

cfg.frameDur = 0.5e-3;
cfg.spec.fs  = 30e6;
cfg.N        = round(cfg.spec.fs * cfg.frameDur);

cfg.spec.nfft  = 256;
cfg.spec.wlen  = 256;
cfg.spec.ovlp  = 192;

cfg.spec.specH = 224;
cfg.spec.specW = 224;

gsFile = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'global_scale.mat');
S = load(gsFile,'best'); best = S.best;
cfg.spec.useFixedGlobalScale = true;
cfg.spec.maxRef_dB   = best.maxRef_dB;
cfg.spec.dynRange_dB = best.dynRange_dB;

cfg.spec.cmapName = "turbo";

cfg.classList = {'LFM','NLFM','FSK','Barker','Frank','P_All','none'};
cfg.maxClassesPerFrame = 3;
cfg.pEmptyFrame = 0.10;

cfg.Fc_range = [-12e6, 12e6];
cfg.minDeltaFc = 3e6;

cfg.PW_range       = [20e-6, 200e-6];
cfg.PRI_range      = [0.2e-3, 10e-3];
cfg.BW_chirp_range = [0.5e6, 2.5e6];
cfg.FSK_spacing_range = [0.1e6, 1e6];

cfg.Costas_M_list  = [6 10 13];
cfg.Barker_lengths = [7 11 13];
cfg.Frank_N_list   = [4 8 16];

cfg.useTaper        = true;
cfg.pulseTaperAlpha = 0.10;

cfg.useChipShaping  = true;
cfg.chipTaperAlpha  = 0.10;

cfg.noiseStdEmpty = 1.0;
end

%% ========================================================================
%  UTILS
% ========================================================================
function val = rand_range(rng2)
val = rng2(1) + (rng2(2) - rng2(1)) * rand;
end

function v = getf(s, f, default)
if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = default; end
end

function w = chip_tukey_weight(tau_chip, Tc, alpha)
alpha = max(0, min(1, alpha));
tau_chip = max(0, min(Tc, tau_chip));

if alpha == 0
    w = ones(size(tau_chip));
    return;
end

edge = (alpha/2) * Tc;
w = ones(size(tau_chip));

i1 = tau_chip < edge;
if any(i1)
    w(i1) = 0.5 * (1 - cos(pi * tau_chip(i1) / edge));
end

i2 = tau_chip > (Tc - edge);
if any(i2)
    w(i2) = 0.5 * (1 - cos(pi * (Tc - tau_chip(i2)) / edge));
end
end
