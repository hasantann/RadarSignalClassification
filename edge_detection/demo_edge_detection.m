function out = demo_edge_detection()
% Gurultu giderilmis spektrogram uzerinde kenar/blob tespiti pipeline demosu.
%
% Tek dosya icinde tum akis:
%   1) Konfigurasyon olustur
%   2) Cok cerceveli (10x) IQ sahne simule et
%   3) 224x2240 gri spektrogram uret (sabit global olcek)
%   4) overlap=0.5 (hop=112) ile yamalara bol
%   5) Denoiser (ve opsiyonel siniflandirici) uygula
%   6) Yamalari ortalama ile yeniden birlestir (mean-stitch)
%   7) Esikleme + baglantili bilesen (CCA) maskesi + sinir kutulari
%
% Not: denNet ve clsNet bulunamazsa demo yine calisir
% (birim donusum denoise + rastgele siniflandirici).

clc; close all; clear;
rng(2)
cfg    = get_dataset_config_v6_short();
snr_dB = 0;

% ---- Model yukleme (proje kokundeki models/ klasoru) ----
projectRoot = fileparts(fileparts(mfilename('fullpath')));
denNet = load(fullfile(projectRoot,'models','denoiser.mat')).bestNet;
clsNet = load(fullfile(projectRoot,'models','classifier.mat')).bestNet;

out = demo_edge_detection_pipeline(cfg, snr_dB, denNet, clsNet);

% Görselleştirme
% figure('Name','Noisy long (gray01)','Position',[100 100 1600 300]);
% imshow(out.ID_long_noisy,[]); title('ID\_long\_noisy (gray01)');
% 
% figure('Name','Denoised long (gray01)','Position',[100 450 1600 300]);
% imshow(out.ID_long_denoised,[]); title('ID\_long\_denoised (gray01)');

disp('Scene-level scores:');
disp(array2table(out.sceneScore,'VariableNames',cfg.classList));

disp('Scene-level preds:');
disp(array2table(out.scenePred,'VariableNames',cfg.classList));

end

%% ========================================================================
%  MAIN PIPELINE
% ========================================================================
function out = demo_edge_detection_pipeline(cfg, snr_dB, denNet, clsNet)

Wlong = 10*cfg.spec.specW;    % 2240
W     = cfg.spec.specW;       % 224
hop   = 224*0.5;                  % overlap 0.5 => 224-112=112 (2. patch 113:336)

% 1) Multi-frame IQ simulate (pulse train)
[iqNoisy, iqClean, meta] = simulate_scene_multiframe_v6(cfg, snr_dB);

% 2) Long spectrogram -> gray01 224x2240 (v6_short ile aynı fixed scale)
[ID_long, ID_clean_long, ~, ~] = iq_to_spec_gray01_v6(iqNoisy, iqClean, cfg, Wlong);

% 3) Overlapped patches
[patches, starts] = slice_overlapped_patches(ID_long, W, hop);

% 4) Denoise + classify (patch-level)
nP = numel(patches);
patchesDen = cell(1,nP);

C = numel(cfg.classList);
scoreMat = zeros(nP, C);

for i=1:nP
    patchesDen{i} = denoise_one_patch(denNet, patches{i});              % 224x224 gray01
    % figure('Name',sprintf('Patch %d: Noisy vs Denoised', i), 'Position',[200 200 600 900]);
    % tl = tiledlayout(2,1,'Padding','compact','TileSpacing','compact');
    % 
    % % ÜST: Noisy
    % nexttile(tl,1);
    % imshow(patches{i}, []);
    % title('Noisy patch');
    % 
    % % ALT: Denoised
    % nexttile(tl,2);
    % imshow(patchesDen{i}, []);
    % title('Denoised patch');
    
    scoreMat(i,:) = classify_multilabel_patch(clsNet, patchesDen{i}, cfg);
end

% 5) Mean stitch -> 224x2240 denoised long
ID_den_long = stitch_mean(patchesDen, starts, Wlong, W);
% =======================
% 6) OTSU (SABİT PARAM)
% =======================
otsuP = struct();
otsuP.k      = 3;   % <-- sabit OTSU çarpanı (istersen 2.75..3.5 dene)
otsuP.alpha  = 40;     % CCA küçük alan eleme (yumuşatılmış)
otsuP.beta1  = 10;     % zaman boşluğu birleştirme (split önleyici)
otsuP.beta2  = 5;      % frekans boşluğu birleştirme
% otsuP.openRad  = 0;    % FP azaltmak istersen 1..2
% otsuP.closeRad = 3;    % disk closing kapalı (line closing kullanacağız)
% otsuP.closeT   = 11;   % <-- YATAY(line) closing: split önleyici (7..21)
% otsuP.minArea  = 50;   % ek küçük eleme (opsiyonel; alpha ile benzer)

[IB_otsu, IC_otsu] = otsu_threshold_and_post(ID_den_long, otsuP);

% =======================
% 7) GÖRSELLEŞTİRME
% =======================
plot_denoised_otsu_ib_ic_stack(ID_den_long, ID_clean_long, ID_long, IB_otsu, IC_otsu);
% İstersen IB'yi de görmek için:
% plot_denoised_otsu_ib_ic_stack(ID_den_long, IB_otsu, IC_otsu);

% =======================
% 7) BOUNDS (OTSU için)
% =======================
[V_otsu, U_otsu] = ic_to_VU(IC_otsu, otsuP.beta1, otsuP.beta2);
B_otsu = masks_to_bounds(V_otsu, U_otsu);
B_otsu.method = "otsu";

% =======================
% 8) Scene-level classification aggregation
% =======================
sceneScore = max(scoreMat,[],1);          % max aggregation
scenePred  = sceneScore > 0.6;

% =======================
% OUT (tek seferde oluştur)
% =======================
out = struct();
out.meta = meta;

out.ID_long_noisy    = ID_long;
out.ID_long_denoised = ID_den_long;

% OTSU sonuçları
out.otsu = struct();
out.otsu.params = otsuP;
out.otsu.IB     = IB_otsu;
out.otsu.IC     = IC_otsu;
out.otsu.V      = V_otsu;
out.otsu.U      = U_otsu;
out.otsu.bounds = B_otsu;

% (İstersen eskisiyle uyum için)
out.bounds = B_otsu;

% classifier çıktıları
out.scoreMat    = scoreMat;
out.sceneScore  = sceneScore;
out.scenePred   = scenePred;
end

%% ========================================================================
%  MULTI-FRAME SCENE SIMULATION (10x shortframe)
% ========================================================================
function [iqNoisy, iqCleanTotal, frameInfo] = simulate_scene_multiframe_v6(cfg, snr_dB)

Nshort = cfg.N;
N      = 10 * Nshort;
fs     = cfg.spec.fs;
t      = (0:N-1).' / fs;

frameDurShort = cfg.frameDur;
frameDur      = 10 * frameDurShort;

iqCleanTotal = complex(zeros(N,1));

% Empty frame?
isEmpty = (rand < cfg.pEmptyFrame);

typePool = {'LFM','NLFM','FSK','Barker','Frank','P_All'};

if isEmpty
    selectedTypes = {};
else
    numTypes = randi([3, cfg.maxClassesPerFrame]);
    perm = randperm(numel(typePool), numTypes);
    selectedTypes = typePool(perm)
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

% Noise (v6_short ile aynı)
refPower   = cfg.sigRefPower;
noisePower = refPower / (10^(snr_dB/10));
noise      = sqrt(noisePower/2) * (randn(N,1) + 1j*randn(N,1));
iqNoisy    = iqCleanTotal + noise;

% frameInfo
frameInfo = struct();
frameInfo.SNR_dB     = snr_dB;
frameInfo.fs         = fs;
frameInfo.N          = N;
frameInfo.frameDur   = frameDur;
frameInfo.numSignals = sigCount;
frameInfo.signals    = signalList;

end

function params = generate_parameters_for_class_multiframe(cfg, className, frameDur)

params = struct();
params.PW  = rand_range(cfg.PW_range);

% PRI: mümkünse >=2 pulse
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

% ToA0
params.ToA0 = rand * min(params.PRI, max(0, frameDur - params.PW));

% numPulses
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
        % ---- FSK tone sequence UPDATED for pulsetrain ----
        % We support:
        %  - Costas: permutation sequence used cyclically across pulses
        %  - NonCostas_Fixed: same random tone every pulse
        %  - NonCostas_RandomEachPulse: new random tone each pulse (optional)
        if rand < 0.5
            params.subtype   = "Costas";
            params.fskMode   = "Costas";
            params.M         = cfg.Costas_M_list(randi(numel(cfg.Costas_M_list)));
            params.tone_spacing = rand_range(cfg.FSK_spacing_range);
            params.BW        = params.M * params.tone_spacing;

            % Costas sequence indices 1..M
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

params.Fc = 0; % dışarıda set edilecek
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

    % pulse index (FSK için)
    p2.pulseIndex = n;

    xTotal = xTotal + simulate_onepulse_by_class(cfg, className, p2, t);
end

end

function x = simulate_onepulse_by_class(cfg, className, params, t)
switch className
    case 'LFM'
        x = simulate_lfm_onepulse(params, t, cfg);

    case 'NLFM'
        x = simulate_nlfm_onepulse(params, t, cfg);

    case 'FSK'
        % FSK mode dispatch
        if isfield(params,'fskMode') && params.fskMode == "Costas"
            x = simulate_costas_onepulse(params, t, cfg);
        else
            x = simulate_fsk_noncostas_onepulse(params, t, cfg);
        end

    case 'Barker'
        x = simulate_barker_onepulse(params, t, cfg);

    case 'Frank'
        x = simulate_frank_onepulse(params, t, cfg);

    case 'P_All'
        switch params.subtype
            case "P1", x = simulate_p1_onepulse(params, t, cfg);
            case "P2", x = simulate_p2_onepulse(params, t, cfg);
            case "P3", x = simulate_p3_onepulse(params, t, cfg);
            case "P4", x = simulate_p4_onepulse(params, t, cfg);
            otherwise, error("Unknown P subtype");
        end

    otherwise
        error("Unknown className: %s", className);
end
end

%% ========================================================================
%  ONE-PULSE GENERATORS (v6_short style)
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

% pulseIndex => seq üzerinde dolaş
if isfield(params,'pulseIndex')
    ii = mod(params.pulseIndex-1, numel(seq)) + 1;
else
    ii = 1;
end
idxTone = seq(ii);          % 1..M

center = (M + 1)/2;
k      = idxTone - center;  % centered index
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

% Fixed tone stored in tone_sequence(1)
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

function x = simulate_p1_onepulse(params, t, cfg), x = simulate_p_polyphase_onepulse(params,t,cfg,"P1"); end
function x = simulate_p2_onepulse(params, t, cfg), x = simulate_p_polyphase_onepulse(params,t,cfg,"P2"); end
function x = simulate_p3_onepulse(params, t, cfg), x = simulate_p_polyphase_onepulse(params,t,cfg,"P3"); end
function x = simulate_p4_onepulse(params, t, cfg), x = simulate_p_polyphase_onepulse(params,t,cfg,"P4"); end

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
% pulse-level taper + power norm (v6_short uyumlu)
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

function [ID_long_gray01, ID_long_clean_gray01, tVec, fVec] = iq_to_spec_gray01_v6(iq, iqClean, cfg, Wtarget)
% Returns:
%   ID_long_gray01: [specH x Wtarget] in [0,1]
%   tVec,fVec: spectrogram axes from compute_spectrogram_fixed

[S_noisy_dB, S_clean_dB, tVec, fVec] = compute_spectrogram_fixed(iq, iqClean, cfg.spec);

maxRef = cfg.spec.maxRef_dB;
minRef = maxRef - cfg.spec.dynRange_dB;
den    = maxRef - minRef;

% NOTE: compute_spectrogram_fixed zaten clip yapacak (tek yerde clip istiyoruz)
% O yüzden burada tekrar clip YAPMIYORUZ. Sadece normalize ediyoruz.
S_norm = (S_noisy_dB - minRef) / max(den, 1e-6);
S_norm = max(min(S_norm,1),0);
ID_long_gray01 = imresize(S_norm, [cfg.spec.specH, Wtarget], "bilinear");

S_norm = (S_clean_dB - minRef) / max(den, 1e-6);
S_norm = max(min(S_norm,1),0);
ID_long_clean_gray01 = imresize(S_norm, [cfg.spec.specH, Wtarget], "bilinear");
end


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
%  OVERLAP PATCHES + DENOISE + STITCH
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

function Y = denoise_one_patch(denNet, X)
% X: 224x224 double [0,1]
% Çıkış: 224x224 double [0,1]

% Network yoksa identity
if isempty(denNet)
    Y = X;
    return;
end

% ---- GİRİŞ ----
X = single(X);                 % 224x224
Xin = repmat(X,1,1,3);         % 224x224x3  (ResNet encoder için)

% GPU'ya taşı (numeric array olarak!)
if canUseGPU
    Xin = gpuArray(Xin);
end

% ---- NETWORK TİPİNE GÖRE ÇAĞRI ----
if isa(denNet,'dlnetwork')
    % === dlnetwork ===
    dlX = dlarray(Xin,"SSCB");
    dlY = predict(denNet, dlX);
    Y   = extractdata(dlY);

elseif isa(denNet,'DAGNetwork') || isa(denNet,'SeriesNetwork')
    % === trainNetwork çıktısı ===
    Y = predict(denNet, Xin);

else
    error('Bilinmeyen network tipi: %s', class(denNet));
end

% ---- ÇIKIŞ DÜZENLEME ----
Y = gather(Y);

% [H W 1] veya [H W 3] gelirse
if ndims(Y) == 3
    Y = Y(:,:,1);
end

Y = max(min(double(Y),1),0);
end


function ID_stitched = stitch_mean(patches, starts, Wlong, W)
H = size(patches{1},1);
acc = zeros(H, Wlong);
wgt = zeros(H, Wlong);

for i = 1:numel(patches)
    s = starts(i);
    e = s + W - 1;
    
    % Toplama işlemi
    acc(:,s:e) = acc(:,s:e) + patches{i};
    wgt(:,s:e) = wgt(:,s:e) + 1;
    
    % Debug: her ekleme sonrası sonucu görselleştir
    % temp_img = acc ./ max(wgt, eps);  % Geçici sonucu hesapla
    % figure;
    % imshow(temp_img, []);
    % title(sprintf('Stitch Step %d', i));  % Başlıkla adım bilgisini göster
end

% Sonuçları döndür
ID_stitched = acc ./ max(wgt, eps);
end


%% ========================================================================
%  CLASSIFICATION (patch-level) -> scores
% ========================================================================
function scores = classify_multilabel_patch(clsNet, patchGray01, cfg)
C = numel(cfg.classList);

% clsNet yoksa random (pipeline kırılmasın)
if isempty(clsNet)
    scores = rand(1,C);
    return;
end

imgRGB = gray01_to_rgb_v6(patchGray01, cfg);  % 224x224x3 uint8
X = im2single(imgRGB);
dlX = dlarray(X,"SSCB");
if canUseGPU, dlX = gpuArray(dlX); end

dlY = forward(clsNet, dlX, "Outputs","sigmoid");

y = gather(extractdata(dlY));
y = squeeze(y);
scores = 1 ./ (1 + exp(-double(y(:).')));  % sigmoid
scores = reshape(scores,1,[]);
if numel(scores) ~= C
    error('Classifier output size mismatch. Expected %d, got %d', C, numel(scores));
end
end

function imgRGB = gray01_to_rgb_v6(gray01, cfg)
switch lower(string(cfg.spec.cmapName))
    case "turbo"
        if exist('turbo','file'), cmap = turbo(256); else, cmap = parula(256); end
    case "jet"
        cmap = jet(256);
    case "hot"
        cmap = hot(256);
    otherwise
        cmap = parula(256);
end

idx = uint16(round(gray01*255)) + 1;
imgRGBd = ind2rgb(idx, cmap);
imgRGB  = im2uint8(imgRGBd);
end
% ---------------- helpers ----------------
function v = make_odd(v)
v = max(3, v);
if mod(v,2)==0, v = v+1; end
end

function B = masks_to_bounds(V, U)
it = find(V>0);
if isempty(it), B.tstart = NaN; B.tstop = NaN;
else, B.tstart = it(1); B.tstop = it(end); end

jf = find(U>0);
if isempty(jf), B.fstart = NaN; B.fstop = NaN;
else, B.fstart = jf(1); B.fstop = jf(end); end
end

%% ========================================================================
%  CFG (v6_short)
% ========================================================================
function cfg = get_dataset_config_v6_short()

cfg.outRoot = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'dataset', 'dataset_multiframe_patches');

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

gsFile = fullfile(fileparts(mfilename('fullpath')), 'global_scale.mat');
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

function v = getf(s, f, default)
if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = default; end
end

function IC = postprocess_IC(IB, p)
% OTSU sonrası split azaltmak için:
% - önce time-line closing
% - sonra opsiyonel open/close (disk)
% - sonra CCA küçük alan eleme
% - sonra V/U gap fill + maskeleme

alpha = p.alpha;
beta1 = p.beta1;
beta2 = p.beta2;

% === (1) TIME-DIRECTION MERGE (split killer) ===
if isfield(p,'closeT') && p.closeT > 0
    closeT = make_odd(round(p.closeT));
    IB = imclose(IB, strel('line', closeT, 0));   % 0 deg => yatay (zaman)
end

% === (2) optional morphology (FP kontrol) ===
if isfield(p,'openRad') && p.openRad > 0
    se = strel('disk', p.openRad);
    IB = imopen(IB, se);
end
if isfield(p,'closeRad') && p.closeRad > 0
    se = strel('disk', p.closeRad);
    IB = imclose(IB, se);
end

% === (3) CCA small component removal ===
CC = bwconncomp(IB, 8);
stats = regionprops(CC, 'Area');
areas = [stats.Area];

IC = IB;

% alpha ile ele
smallIdx = find(areas < alpha);
for idx = smallIdx
    IC(CC.PixelIdxList{idx}) = 0;
end

% ekstra minArea (opsiyonel)
if isfield(p,'minArea') && p.minArea > 0
    CC2 = bwconncomp(IC, 8);
    st2 = regionprops(CC2,'Area');
    ar2 = [st2.Area];
    small2 = find(ar2 < p.minArea);
    for idx = small2
        IC(CC2.PixelIdxList{idx}) = 0;
    end
end

[F,~] = size(IC);

% === (4) time mask V + gap fill ===
V = any(IC,1);
S = find(V==1);
if numel(S)>=2
    for i=1:numel(S)-1
        if S(i+1)-S(i) < beta1
            V(S(i):S(i+1)) = 1;
        end
    end
end

% === (5) freq mask U + gap fill ===
U = false(F,1);
idxT = find(V==1);
if ~isempty(idxT)
    rowSum = sum(IC(:,idxT),2);
    U(rowSum>0) = true;
end
K = find(U==1);
if numel(K)>=2
    for j=1:numel(K)-1
        if K(j+1)-K(j) < beta2
            U(K(j):K(j+1)) = 1;
        end
    end
end

% === (6) apply masks ===
IC = IC & (U * V);
end

function [IB, IC] = otsu_threshold_and_post(ID_gray01, p)
% ID_gray01 : [F x T] double [0,1]
% p.k       : OTSU multiplier
% p.alpha,beta1,beta2, openRad, closeRad, closeT, minArea

% 1) OTSU threshold
T0 = graythresh(ID_gray01) * p.k;
IB = imbinarize(ID_gray01, T0);

% 2) Postprocess -> IC
IC = postprocess_IC(IB, p);

end

function plot_denoised_otsu_ib_ic_stack(ID_denoised, ID_clean, ID_noisy ,IB_otsu, IC_otsu)
% İstersen 3 satır: denoised / IB / IC

% Ekran boyutunu al (primary monitor)
scr = get(0,'ScreenSize');   % [left bottom width height]

% Ekranın yarısı kadar boyut
w = round(scr(3) * 0.50);
h = round(scr(4) * 0.80);

% Ortala
x = scr(1) + round((scr(3) - w)/2);
y = scr(2) + round((scr(4) - h)/2);

figure('Name',"Denoised vs OTSU(IB/IC)", ...
       'WindowStyle',"alwaysontop", ...
       'Units','pixels', ...
       'Position',[x y w h]);

tl = tiledlayout(5,1,"padding","compact","tilespacing","compact");

nexttile(tl);
imshow(ID_clean, []);
title('Clean (gray01)');

nexttile(tl);
imshow(ID_noisy, []);
title('Noisy (gray01)');

nexttile(tl);
imshow(ID_denoised, []);
title('Denoised stitched (gray01)');

nexttile(tl);
imshow(IB_otsu, []);
title('OTSU (IB)');

nexttile(tl);
imshow(IC_otsu, []);
title('OTSU + postprocess (IC)');
end

function [V,U] = ic_to_VU(IC, beta1, beta2)
[F,T] = size(IC);

V = zeros(1,T);
for tt=1:T
    if any(IC(:,tt)), V(tt)=1; end
end
S = find(V==1);
if numel(S)>=2
    for i=1:numel(S)-1
        if S(i+1)-S(i) < beta1
            V(S(i):S(i+1)) = 1;
        end
    end
end

U = zeros(1,F);
idxT = find(V==1);
if ~isempty(idxT)
    rowSum = sum(IC(:,idxT),2);
    U(rowSum>0) = 1;
end
K = find(U==1);
if numel(K)>=2
    for j=1:numel(K)-1
        if K(j+1)-K(j) < beta2
            U(K(j):K(j+1)) = 1;
        end
    end
end
end

function rdw = demo_edge_detection_wrapper_rdw(cfg, snr_dB, denNet, clsNet)
% Kenar tespiti pipeline'ini calistirir ve uzerine RDW cikarimi ekler:
%   1) demo_edge_detection_pipeline calistirilir
%   2) Denoised uzun spektrogram + IC uzerinden RDW/cluster/PRI cikarilir
%   3) Spektrogram uzerine renkli sinir kutulari cizilir

% --- 1) pipeline ciktisi ---
outDemo = demo_edge_detection_pipeline(cfg, snr_dB, denNet, clsNet);

ID_den = outDemo.ID_long_denoised;     % 224x2240 gray01
ID_nsy = outDemo.ID_long_noisy;
ID_cln = [];
if isfield(outDemo,'ID_long_clean'), ID_cln = outDemo.ID_long_clean; end %#ok<NASGU>

% OTSU-IC hazırsa onu kullan, yoksa RDW fonksiyonu kendi OTSU'sunu yapar
IC = [];
if isfield(outDemo,'otsu') && isfield(outDemo.otsu,'IC')
    IC = outDemo.otsu.IC;
end

% tVec/fVec demo tarafinda uretilmiyorsa RDW bin-ekseni ile calisir.
% Gerekirse burada zaman/frekans eksenleri uretilip gonderilebilir.
tVec = [];
fVec = [];

% --- 2) RDW extraction ---
if isempty(IC)
    rdw = radar_descriptive_word(ID_den, ...
        'OtsuK', 3, 'Alpha', 40, 'BetaT', 10, 'BetaF', 5, 'CloseT', 11, ...
        'TolPW', 10, 'TolPRI', 12, 'TolBW', 20, 'TolFc', 25, ...
        'FcHopStd', 30, 'PerScoreMin', 0.55, ...
        'tVec', tVec, 'fVec', fVec);
else
    rdw = radar_descriptive_word(ID_den, ...
        'IC', IC, ...
        'TolPW', 10, 'TolPRI', 12, 'TolBW', 20, 'TolFc', 25, ...
        'FcHopStd', 30, 'PerScoreMin', 0.55, ...
        'tVec', tVec, 'fVec', fVec);
end

% Demo çıktısını da içine gömelim (kolay debug)
rdw.demo = outDemo;

% --- 3) Plot: bbox on spectrogram (cluster colors) ---
plot_events_bboxes(ID_den, rdw.events, rdw.eventClusterId, rdw.clusters, ...
    'FigureName', 'Denoised + Event BBoxes (cluster-colored)', ...
    'ShowText', true);

% İstersen noisy üzerine de çiz
plot_events_bboxes(ID_nsy, rdw.events, rdw.eventClusterId, rdw.clusters, ...
    'FigureName', 'Noisy + Event BBoxes (cluster-colored)', ...
    'ShowText', false);
end
