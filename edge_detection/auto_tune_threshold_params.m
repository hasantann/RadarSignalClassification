function best = auto_tune_threshold_params(ID_gray01, tVec, fVec, meta, cfg, opts)
% Esikleme + son-isleme parametrelerini yer gercegine (GT) gore otomatik ayarlar.
%
% ID_gray01 : [H x W] double, [0,1] (denoised uzun spektrogram)
% tVec      : STFT zaman ekseni (sn), uzunluk T0
% fVec      : STFT frekans ekseni (Hz), uzunluk F0
% meta      : sahne simulasyonu ciktisi (meta.signals(k).ToA_list, parametreler)
% cfg       : konfigurasyon (opsiyonel)
% opts      : ayar secenekleri
%
% Output:
%   best.(method).params  : tuned threshold+post params for that method
%   best.(method).score   : best score
%   best.post.params      : optional global post params tuned across methods (tunePost=true)
%   best.GT               : resized GT mask to [H x W]

if nargin < 6, opts = struct(); end
opts = set_default_opts_fp(opts);

H = size(ID_gray01,1);
W = size(ID_gray01,2);

% 1) GT mask on STFT grid [F0 x T0]
GT0 = build_gt_mask_stft(meta, tVec, fVec, cfg, opts);  % logical [F0 x T0]

% 2) Resize GT to image grid [H x W]
GT = imresize(GT0, [H W], 'nearest');
GT = logical(GT);

best = struct();
best.GT = GT;

methodList = ["otsu","adaptive_mean","adaptive_gaussian","median_dynamic","snr_local"];

% 3) Method-wise tuning (threshold + post + optional morphology)
for mi = 1:numel(methodList)
    method = methodList(mi);

    cand = build_candidate_grid(method, opts);     % table with right var names
    [pBest, sBest] = search_best_params(ID_gray01, GT, method, cand, opts);

    best.(method).params = pBest;
    best.(method).score  = sBest;
end

% 4) Optionally tune ONLY post params globally across methods (keeps each method's threshold fixed)
if opts.tunePost
    postCand = build_post_candidate_grid(opts); % table(alpha,beta1,beta2,morph)
    [postBest, postScore] = search_best_post(ID_gray01, GT, best, postCand, opts);
    best.post.params = postBest;
    best.post.score  = postScore;
end

end

%% ========================= DEFAULT OPTS (FP-ODAKLI) =========================
function opts = set_default_opts_fp(opts)

% ----- Scoring / constraints -----
opts.beta         = getf(opts,'beta', 0.3);     % <1 => precision ağırlıklı (FP düşürür)
opts.fpWeight     = getf(opts,'fpWeight', 3.0); % FP cezası (arttır => daha az false alarm)
opts.minRecall    = getf(opts,'minRecall', 0.25); % recall çok düşmesin (0.2-0.4 iyi)
opts.useIC        = getf(opts,'useIC', true);   % IC üzerinden optimize
opts.maxEval      = getf(opts,'maxEval', 400);  % method başına max deneme
opts.randomSeed   = getf(opts,'randomSeed', 0);
opts.debug        = getf(opts,'debug', false);

% ----- GT tolerans -----
opts.timePadBins  = getf(opts,'timePadBins', 1);
opts.freqPadBins  = getf(opts,'freqPadBins', 1);

% ----- Post grid (CCA + gap fill) -----
opts.tunePost     = getf(opts,'tunePost', true);
opts.alphaGrid    = getf(opts,'alphaGrid', [40 60 80 100 140 180]);   % küçük objeleri sil (FP düşürür)
opts.beta1Grid    = getf(opts,'beta1Grid', [2 4 6 8 10 12]);          % time gap fill
opts.beta2Grid    = getf(opts,'beta2Grid', [1 2 3 4 5 6]);            % freq gap fill

% ----- Morphology (FP düşürmek için çok etkili) -----
opts.useMorph     = getf(opts,'useMorph', true);
opts.minAreaGrid  = getf(opts,'minAreaGrid', [0 20 40 60 80 120]);     % bwareaopen
opts.openRadGrid  = getf(opts,'openRadGrid', [0 1 2]);                 % imopen disk radius
opts.closeRadGrid = getf(opts,'closeRadGrid',[0 1 2 3]);               % imclose disk radius

end

function v = getf(s,f,d)
if isfield(s,f) && ~isempty(s.(f)), v = s.(f); else, v = d; end
end

%% ========================= GT MASK (STFT GRID) =========================
function GT = build_gt_mask_stft(meta, tVec, fVec, cfg, opts)
% GT: [numel(fVec) x numel(tVec)] logical
F0 = numel(fVec);
T0 = numel(tVec);
GT = false(F0, T0);

if ~isfield(meta,'signals') || isempty(meta.signals)
    return;
end

for k = 1:numel(meta.signals)
    if ~isfield(meta.signals(k),'parameters'), continue; end
    p = meta.signals(k).parameters;

    % ToA list
    if isfield(meta.signals(k),'ToA_list') && ~isempty(meta.signals(k).ToA_list)
        ToAs = meta.signals(k).ToA_list(:);
    else
        if isfield(p,'ToA0') && isfield(p,'PRI') && isfield(p,'numPulses')
            ToAs = p.ToA0 + (0:max(p.numPulses-1,0))*p.PRI;
            ToAs = ToAs(:);
        elseif isfield(p,'ToA')
            ToAs = p.ToA(:);
        else
            continue;
        end
    end

    if ~isfield(p,'PW') || ~isfield(p,'Fc'), continue; end
    PW = p.PW;
    Fc = p.Fc;

    if isfield(p,'BW') && ~isempty(p.BW)
        BW = p.BW;
    else
        BW = 1e6; % fallback
    end

    f1 = Fc - BW/2;
    f2 = Fc + BW/2;

    fi = (fVec >= min(f1,f2)) & (fVec <= max(f1,f2));
    if ~any(fi), continue; end

    for n = 1:numel(ToAs)
        t1 = ToAs(n);
        t2 = ToAs(n) + PW;

        ti = (tVec >= t1) & (tVec <= t2);
        if ~any(ti), continue; end

        GT(fi, ti) = true;
    end
end

% ---- pad GT (tolerans) ----
GT = pad_gt(GT, opts.freqPadBins, opts.timePadBins);

end

function GT2 = pad_gt(GT, padF, padT)
GT2 = GT;
if padF > 0
    GT2 = imdilate(GT2, strel('line', 2*padF+1, 90)); % vertical grow
end
if padT > 0
    GT2 = imdilate(GT2, strel('line', 2*padT+1, 0));  % horizontal grow
end
end

%% ========================= CANDIDATE GRID (METHOD + POST + MORPH) =========================
function cand = build_candidate_grid(method, opts)
rng(opts.randomSeed);

% --- method-specific ---
switch string(method)

    case "otsu"
        % k: multiplier
        kGrid = (1.0:0.25:7.0)';
        base = table(kGrid, 'VariableNames',{'k'});

    case {"adaptive_mean","adaptive_gaussian"}
        sensGrid = (0.25:0.05:0.85)';
        nbGrid   = [15 21 25 31 35 41 51]';   % odd
        [S,N] = ndgrid(sensGrid, nbGrid);
        base = table(S(:), N(:), 'VariableNames',{'sens','nb'});

    case "median_dynamic"
        medGrid = [11 15 21 25 31 41 51]';
        kGrid   = (0.8:0.2:4.0)';
        [M,K] = ndgrid(medGrid, kGrid);
        base = table(M(:), K(:), 'VariableNames',{'med','k'});

    case "snr_local"
        pctlGrid = [45 50 55 60 65 70]';
        k2Grid   = (1.0:0.1:2.6)';
        [P,K2] = ndgrid(pctlGrid, k2Grid);
        base = table(P(:), K2(:), 'VariableNames',{'pctl','k2'});

    otherwise
        error("Unknown method: %s", method);
end

% --- post grid sample (küçük tut) ---
[A,B1,B2] = ndgrid(opts.alphaGrid(:), opts.beta1Grid(:), opts.beta2Grid(:));
postAll = [A(:) B1(:) B2(:)];
nPost = min(size(postAll,1), 30);          % büyükse sample
idx = randperm(size(postAll,1), nPost);
postAll = postAll(idx,:);

% --- morph grid sample (FP düşürür) ---
if opts.useMorph
    [MA,OR,CR] = ndgrid(opts.minAreaGrid(:), opts.openRadGrid(:), opts.closeRadGrid(:));
    morphAll = [MA(:) OR(:) CR(:)];
    nMorph = min(size(morphAll,1), 30);
    idxm = randperm(size(morphAll,1), nMorph);
    morphAll = morphAll(idxm,:);
else
    morphAll = [0 0 0];
end

% --- expand base x post x morph ---
cand = table();
for i=1:height(base)
    rowB = base(i,:);
    for j=1:size(postAll,1)
        for k=1:size(morphAll,1)
            row = rowB;

            row.alpha = postAll(j,1);
            row.beta1 = postAll(j,2);
            row.beta2 = postAll(j,3);

            row.minArea  = morphAll(k,1);
            row.openRad  = morphAll(k,2);
            row.closeRad = morphAll(k,3);

            cand = [cand; row]; %#ok<AGROW>
        end
    end
end

% shuffle and cap by opts.maxEval (search_best_params zaten kırpıyor)
cand = cand(randperm(height(cand)), :);

end

function postCand = build_post_candidate_grid(opts)
[A,B1,B2] = ndgrid(opts.alphaGrid(:), opts.beta1Grid(:), opts.beta2Grid(:));

if opts.useMorph
    [MA,OR,CR] = ndgrid(opts.minAreaGrid(:), opts.openRadGrid(:), opts.closeRadGrid(:));
    postCand = table();
    for i=1:numel(A)
        for k=1:numel(MA)
            postCand = [postCand; table(A(i),B1(i),B2(i),MA(k),OR(k),CR(k), ...
                'VariableNames',{'alpha','beta1','beta2','minArea','openRad','closeRad'})]; %#ok<AGROW>
        end
    end
else
    postCand = table(A(:),B1(:),B2(:), zeros(numel(A),1),zeros(numel(A),1),zeros(numel(A),1), ...
        'VariableNames',{'alpha','beta1','beta2','minArea','openRad','closeRad'});
end

end

%% ========================= SEARCH BEST (METHOD) =========================
function [pBest, sBest] = search_best_params(ID, GT, method, cand, opts)

sBest = -inf;
pBest = struct();

nTry = min(height(cand), opts.maxEval);

for i=1:nTry
    p = table2struct(cand(i,:));

    % 1) threshold
    IB = apply_threshold_method(ID, method, p);

    % 2) post + morph
    if opts.useIC
        pred = postprocess_IC(IB, p, opts);
    else
        pred = IB;
        if opts.useMorph
            pred = morph_cleanup(pred, p);
        end
    end

    % 3) scoring (FP penalized)
    [sc, prec, rec, fpRatio] = score_mask_fp_penalized(pred, GT, opts);

    if opts.debug && mod(i,50)==0
        fprintf('[%s] try %d/%d  sc=%.4f prec=%.3f rec=%.3f fpR=%.3f\n', method, i,nTry, sc, prec, rec, fpRatio);
    end

    if sc > sBest
        sBest = sc;
        pBest = p;
        pBest.score     = sc;
        pBest.precision = prec;
        pBest.recall    = rec;
        pBest.fpRatio   = fpRatio;
    end
end

end

%% ========================= APPLY THRESHOLD METHODS =========================
function IB = apply_threshold_method(ID, method, p)
% Robust getter (NO struct-field arithmetic hack)
getf = @(s, f, d) (isfield(s,f) && ~isempty(s.(f))) * true;

switch string(method)

    case "otsu"
        if ~isfield(p,'k') || isempty(p.k), p.k = 3.5; end
        T0 = graythresh(ID) * p.k;
        IB = imbinarize(ID, T0);

    case "adaptive_mean"
        if ~isfield(p,'sens') || isempty(p.sens), p.sens = 0.7; end
        if ~isfield(p,'nb')   || isempty(p.nb),   p.nb   = 25;  end
        nb = make_odd(round(p.nb));
        T1 = adaptthresh(ID, p.sens, 'NeighborhoodSize',[nb nb], 'Statistic','mean');
        IB = imbinarize(ID, T1);

    case "adaptive_gaussian"
        if ~isfield(p,'sens') || isempty(p.sens), p.sens = 0.5; end
        if ~isfield(p,'nb')   || isempty(p.nb),   p.nb   = 35;  end
        nb = make_odd(round(p.nb));
        T2 = adaptthresh(ID, p.sens, 'NeighborhoodSize',[nb nb], 'Statistic','gaussian');
        IB = imbinarize(ID, T2);

    case "median_dynamic"
        if ~isfield(p,'med') || isempty(p.med), p.med = 25; end
        if ~isfield(p,'k')   || isempty(p.k),   p.k   = 2.0; end
        medw = make_odd(round(p.med));

        noise = medfilt2(ID, [medw medw]);
        diffv = ID - noise;
        T3    = noise + p.k * std(diffv(:));
        IB    = ID > T3;

    case "snr_local"
        if ~isfield(p,'pctl') || isempty(p.pctl), p.pctl = 60; end
        if ~isfield(p,'k2')   || isempty(p.k2),   p.k2   = 1.5; end
        noise_col = prctile(ID, p.pctl, 1);

        IB = false(size(ID));
        for tt=1:size(ID,2)
            IB(:,tt) = ID(:,tt) > noise_col(tt) * p.k2;
        end

    otherwise
        error("Unknown method: %s", method);
end
end

function v = make_odd(v)
v = max(3, v);
if mod(v,2)==0, v = v+1; end
end

%% ========================= POSTPROCESS (CCA + GAP FILL + MORPH) =========================
function IC = postprocess_IC(IB, p, opts)

% required fields
if ~isfield(p,'alpha') || isempty(p.alpha), p.alpha = 60; end
if ~isfield(p,'beta1') || isempty(p.beta1), p.beta1 = 10; end
if ~isfield(p,'beta2') || isempty(p.beta2), p.beta2 = 3;  end

alpha = p.alpha;
beta1 = p.beta1;
beta2 = p.beta2;

% 1) remove small CC by Area
CC = bwconncomp(IB, 8);
if CC.NumObjects == 0
    IC = false(size(IB));
    return;
end
stats = regionprops(CC, 'Area');
areas = [stats.Area];

IC = IB;
smallIdx = find(areas < alpha);
for idx = smallIdx
    IC(CC.PixelIdxList{idx}) = 0;
end

% 2) build V/U masks (gap fill)
[F,T] = size(IC);

V = any(IC,1);
V = fill_gaps_1d(V, beta1);

U = false(F,1);
idxT = find(V);
if ~isempty(idxT)
    rowSum = sum(IC(:,idxT),2);
    U(rowSum>0) = true;
end
U = fill_gaps_1d(U(:).', beta2);
U = U(:);

% 3) mask by U,V
IC = IC & (U * V);

% 4) morphology cleanup (FP düşürür)
if opts.useMorph
    IC = morph_cleanup(IC, p);
end

end

function x = fill_gaps_1d(x, gap)
x = logical(x(:).');
idx = find(x);
if numel(idx) >= 2
    for i=1:numel(idx)-1
        if (idx(i+1) - idx(i)) <= gap
            x(idx(i):idx(i+1)) = true;
        end
    end
end
end

function BW = morph_cleanup(BW, p)
BW = logical(BW);

if ~isfield(p,'minArea') || isempty(p.minArea), p.minArea = 0; end
if ~isfield(p,'openRad') || isempty(p.openRad), p.openRad = 0; end
if ~isfield(p,'closeRad')|| isempty(p.closeRad),p.closeRad= 0; end

% remove tiny speckles
if p.minArea > 0
    BW = bwareaopen(BW, round(p.minArea), 8);
end

% open (remove thin noise)
if p.openRad > 0
    se = strel('disk', round(p.openRad), 0);
    BW = imopen(BW, se);
end

% close (fill small holes)
if p.closeRad > 0
    se = strel('disk', round(p.closeRad), 0);
    BW = imclose(BW, se);
end
end

%% ========================= SCORING (FP PENALIZED) =========================
function [sc, prec, rec, fpRatio] = score_mask_fp_penalized(pred, GT, opts)

pred = logical(pred);
GT   = logical(GT);

% size safety
if ~isequal(size(pred), size(GT))
    GT = imresize(GT, size(pred), 'nearest');
    GT = logical(GT);
end

TP = nnz(pred & GT);
FP = nnz(pred & ~GT);
FN = nnz(~pred & GT);

prec = TP / max(TP + FP, 1);
rec  = TP / max(TP + FN, 1);

% precision-weighted F_beta
b = opts.beta;
F = (1+b^2) * (prec*rec) / max((b^2)*prec + rec, eps);

% false alarm penalty (normalize by GT size)
fpRatio = FP / max(nnz(GT), 1);
penFP   = opts.fpWeight * fpRatio;

sc = F - penFP;

% recall guard
if rec < opts.minRecall
    % recall düşerse sert ceza (FP düşürürken sinyali öldürmesin)
    sc = sc - 1.0 * (opts.minRecall - rec) / max(opts.minRecall, 1e-6);
end

end

%% ========================= GLOBAL POST TUNING (OPTIONAL) =========================
function [postBest, sBest] = search_best_post(ID, GT, best, postCand, opts)

methods = ["otsu","adaptive_mean","adaptive_gaussian","median_dynamic","snr_local"];

sBest = -inf;
postBest = struct();

nTry = min(height(postCand), 600);

for i=1:nTry
    pp = table2struct(postCand(i,:));

    scores = zeros(1,numel(methods));

    for m=1:numel(methods)
        method = methods(m);
        if ~isfield(best, method)
            scores(m) = -inf;
            continue;
        end

        p = best.(method).params;

        % override post+morph
        p.alpha    = pp.alpha;
        p.beta1    = pp.beta1;
        p.beta2    = pp.beta2;
        p.minArea  = pp.minArea;
        p.openRad  = pp.openRad;
        p.closeRad = pp.closeRad;

        IB = apply_threshold_method(ID, method, p);
        IC = postprocess_IC(IB, p, opts);

        [sc] = score_mask_fp_penalized(IC, GT, opts);
        scores(m) = sc;
    end

    scAll = mean(scores(isfinite(scores)));

    if scAll > sBest
        sBest = scAll;
        postBest = pp;
        postBest.score = scAll;
        postBest.scoresPerMethod = scores;
    end
end

end
