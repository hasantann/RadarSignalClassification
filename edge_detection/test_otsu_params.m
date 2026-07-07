function results = test_otsu_params()
% Otsu esikleme parametrelerini hizlica tarar/kiyaslar.
% Girdi olarak kenar tespiti demosunun uzun denoised spektrogramini kullanir.
% Cikti: results (parametre tablosu + en iyi adaylar)

clc; close all;

% 0) Demo ciktisi
out = demo_edge_detection();
ID  = out.ID_long_denoised;   % [224 x 2240] gray01 double

% =========================================================
% 1) Test edilecek parametre setleri (istersen genişlet)
% =========================================================
K_list      = [2.75 3.0 3.25 3.5];
alpha_list  = [40 80 120];
beta1_list  = [6 10 14];
beta2_list  = [3 5 7];
open_list   = [0 1];          % FP azaltma
closeT_list = [7 11 15 21];   % split azaltma
minArea_list= [0 120 200];    % ekstra eleme

% base
base = struct();
base.k        = 3.0;
base.alpha    = 80;
base.beta1    = 10;
base.beta2    = 5;
base.openRad  = 0;
base.closeRad = 0;   % disk closing'i kapalı tut (line closing yeter)
base.closeT   = 11;
base.minArea  = 120;

% =========================================================
% 2) Grid search
% =========================================================
cnt = 0;
rows = [];

for k = K_list
for a = alpha_list
for b1 = beta1_list
for b2 = beta2_list
for op = open_list
for ct = closeT_list
for ma = minArea_list

    p = base;
    p.k       = k;
    p.alpha   = a;
    p.beta1   = b1;
    p.beta2   = b2;
    p.openRad = op;
    p.closeT  = ct;
    p.minArea = ma;

    [IB, IC] = otsu_threshold_and_post(ID, p);

    % ---- basit metrikler (ground-truth yokken) ----
    % (1) aktif piksel oranı: çok küçükse kaçırıyor, çok büyükse FP
    onRatio = nnz(IC) / numel(IC);

    % (2) component sayısı: split/FP göstergesi
    CC = bwconncomp(IC, 8);
    nCC = CC.NumObjects;

    % (3) en büyük component oranı: tek blob'a mı gidiyor?
    if nCC > 0
        areas = cellfun(@numel, CC.PixelIdxList);
        maxAreaRatio = max(areas) / numel(IC);
        medArea = median(areas);
    else
        maxAreaRatio = 0;
        medArea = 0;
    end

    % (4) V/U mask genişliği (kaplama) – "ne kadar süre ve bantta var"
    V = any(IC,1); U = any(IC,2);
    vRatio = mean(V);
    uRatio = mean(U);

    cnt = cnt + 1;
    rows(cnt,:) = [k a b1 b2 op ct ma onRatio nCC maxAreaRatio medArea vRatio uRatio]; %#ok<AGROW>

end
end
end
end
end
end
end

vars = ["k","alpha","beta1","beta2","openRad","closeT","minArea", ...
        "onRatio","nCC","maxAreaRatio","medArea","vRatio","uRatio"];
T = array2table(rows,'VariableNames',vars);

% =========================================================
% 3) "İyi" adayları seçmek için basit skor (heuristic)
%    - onRatio hedef bandı: 0.1%..5% (dataset'e göre değişir)
%    - nCC düşük olsun (split/FP azalır)
%    - maxAreaRatio çok yüksekse tek dev blob (istenmeyebilir)
% =========================================================
onTarget = clamp01( 1 - abs(log10(max(T.onRatio,1e-9)) - log10(0.01)) / 3 ); % 1% civarı iyi
ccTarget = clamp01( 1 ./ (1 + 0.05*T.nCC) );
blobPen  = clamp01( 1 - max(0, T.maxAreaRatio - 0.25) / 0.75 ); % >25% kaplıyorsa penalize

T.score = 0.45*onTarget + 0.45*ccTarget + 0.10*blobPen;

T = sortrows(T, "score", "descend");

% en iyi ilk 10'u yaz
disp(T(1:min(10,height(T)), :));

results = struct();
results.table = T;

% =========================================================
% 4) En iyi ilk N adayın görsel kıyası
% =========================================================
Nshow = min(6, height(T));
figure('Name',"Top candidates (IC)", 'Position',[50 80 1600 900]);
tl = tiledlayout(2,3,'Padding','compact','TileSpacing','compact');

for i=1:Nshow
    nexttile(tl,i);
    p = tableRowToParam(T(i,:), base);
    [~, IC] = otsu_threshold_and_post(ID, p);
    imshow(IC, []);
    title(sprintf("score=%.3f | k=%.2f a=%d b1=%d b2=%d op=%d ct=%d ma=%d", ...
        T.score(i), T.k(i), T.alpha(i), T.beta1(i), T.beta2(i), ...
        T.openRad(i), T.closeT(i), T.minArea(i)));
end

% En iyisini ayrıca stacked göster (denoised / IB / IC)
bestP = tableRowToParam(T(1,:), base);
[IBbest, ICbest] = otsu_threshold_and_post(ID, bestP);
plot_denoised_otsu_ib_ic_stack(ID, ID, ID, IBbest, ICbest); % clean/noisy yoksa aynı gösterir
sgtitle("Best param set – quick check");

end

% ---------------- small helpers ----------------
function p = tableRowToParam(row, base)
p = base;
p.k       = row.k;
p.alpha   = row.alpha;
p.beta1   = row.beta1;
p.beta2   = row.beta2;
p.openRad = row.openRad;
p.closeT  = row.closeT;
p.minArea = row.minArea;
end

function y = clamp01(x)
y = max(0, min(1, x));
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

function plot_denoised_and_otsu_stack(ID_denoised, IC_otsu)
% 1 sütun, 2 satır: üst denoised, alt OTSU(IC)

figure('Name',"Denoised vs OTSU(IC)", 'Position',[200 80 900 900]);
tl = tiledlayout(2,1,"padding","compact","tilespacing","compact");

nexttile(tl);
imshow(ID_denoised, []);
title('Denoised stitched (gray01)');

nexttile(tl);
imshow(IC_otsu, []);
title('OTSU + postprocess (IC)');
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