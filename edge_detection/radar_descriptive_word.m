function out = radar_descriptive_word(ID_gray01, varargin)
% Gurultu giderilmis spektrogramdan radar tanimlayici kelime (RDW) cikarimi.
%
% Kenar/blob tespiti -> olay tablosu -> PRI/PW/BW/Fc kumeleme -> sinif etiketi.
% Dis bagimlilik yoktur (MATLAB built-in + Image Processing Toolbox).
%
% GIRDI:
%   ID_gray01 : [F x T] double, [0,1] (birlestirilmis denoised spektrogram)
%
% OPSİYONEL (Name-Value):
%   'IC'        : hazır IC mask (logical [F x T]) -> OTSU yapılmaz
%   'tVec'      : spectrogram zaman ekseni (1xT veya Tx1)
%   'fVec'      : spectrogram frekans ekseni (Fx1 veya 1xF)
%   'OtsuK'     : OTSU multiplier (default 3)
%   'Alpha'     : küçük bileşen eleme (default 40)
%   'BetaT'     : V gap fill (time) (default 10)
%   'BetaF'     : U gap fill (freq) (default 5)
%   'CloseT'    : yatay line closing (split önleyici) (default 11)
%   'MergeGapT' : blob->event merge time gap (bin) (default 6)
%   'MergePwTol': blob->event merge PW tol (bin) (default 6)
%   'MinArea'   : min area (default 0; alpha zaten var)
%
%   'TolPW'     : cluster PW toleransı (bin) (default 10)
%   'TolPRI'    : cluster PRI toleransı (bin) (default 12)
%   'TolBW'     : cluster BW toleransı (bin) (default 20)
%   'TolFc'     : cluster Fc toleransı (bin) (default 25)  % FSK'de yumuşatılır
%   'FcHopStd'  : FSK adaylığı için Fc std eşiği (bin) (default 30)
%   'PerScoreMin' : periyodiklik skoru min (0..1) (default 0.55)
%
% ÇIKTI:
%   out.IC, out.IB, out.events (table), out.clusters (struct array)
%   out.eventClusterId, out.clusterWord, out.clusterLabel, out.clusterPRI
%   out.debug (fig handles vs.)
%
% Notlar:
% - PRI tüm sınıflarda sabit varsayımı: cluster kararında PRI/periodicity ağırlığı yüksek.
% - FSK: Fc aynı kalmak zorunda değil -> PRI+PW ile bağlanır, Fc sadece yardımcı.
%
% Hasan Tan - all-in-one

%% ---------------- Parse opts ----------------
p = inputParser;
p.addRequired('ID_gray01', @(x) isnumeric(x) && ndims(x)==2);

p.addParameter('IC', [], @(x) isempty(x) || islogical(x));
p.addParameter('tVec', [], @(x) isempty(x) || isnumeric(x));
p.addParameter('fVec', [], @(x) isempty(x) || isnumeric(x));

% OTSU+post
p.addParameter('OtsuK', 3, @(x) isnumeric(x) && isscalar(x) && x>0);
p.addParameter('Alpha', 40, @(x) isnumeric(x) && isscalar(x));
p.addParameter('BetaT', 10, @(x) isnumeric(x) && isscalar(x));
p.addParameter('BetaF', 5,  @(x) isnumeric(x) && isscalar(x));
p.addParameter('CloseT', 11, @(x) isnumeric(x) && isscalar(x));

p.addParameter('MergeGapT', 6, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('MergePwTol', 6, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('MinArea', 0, @(x) isnumeric(x) && isscalar(x) && x>=0);

% clustering
p.addParameter('TolPW', 10, @(x) isnumeric(x) && isscalar(x) && x>0);
p.addParameter('TolPRI', 12, @(x) isnumeric(x) && isscalar(x) && x>0);
p.addParameter('TolBW', 20, @(x) isnumeric(x) && isscalar(x) && x>0);
p.addParameter('TolFc', 25, @(x) isnumeric(x) && isscalar(x) && x>0);

p.addParameter('FcHopStd', 30, @(x) isnumeric(x) && isscalar(x) && x>=0);
p.addParameter('PerScoreMin', 0.55, @(x) isnumeric(x) && isscalar(x) && x>=0 && x<=1);

p.parse(ID_gray01, varargin{:});
opt = p.Results;

ID = double(ID_gray01);
ID = max(min(ID,1),0);
[F,T] = size(ID);

tVec = opt.tVec;
fVec = opt.fVec;
if isempty(tVec), tVec = 1:T; end
if isempty(fVec), fVec = 1:F; end
tVec = tVec(:).';    % 1xT
fVec = fVec(:);      % Fx1

%% ---------------- 1) IB/IC ----------------
if isempty(opt.IC)
    pp = struct();
    pp.k      = opt.OtsuK;
    pp.alpha  = opt.Alpha;
    pp.beta1  = opt.BetaT;
    pp.beta2  = opt.BetaF;
    pp.closeT = opt.CloseT;
    pp.minArea = opt.MinArea;
    [IB, IC] = otsu_threshold_and_post(ID, pp);
else
    IC = opt.IC;
    IB = [];
end

%% ---------------- 2) IC -> raw blobs ----------------
blobs = extract_blobs_from_IC(IC);

%% ---------------- 3) blobs -> pulse events (merge split) ----------------
events = blobs_to_events_merge(blobs, opt.MergeGapT, opt.MergePwTol);

if isempty(events)
    out = struct();
    out.IB = IB;
    out.IC = IC;
    out.events = table();
    out.clusters = struct([]);
    out.eventClusterId = [];
    out.clusterWord = {};
    out.clusterLabel = {};
    out.clusterPRI = [];
    return;
end

%% ---------------- 4) event params (bin + physical) ----------------
evTbl = events_to_table(events, tVec, fVec);

%% ---------------- 5) Local PRI estimate per event (PW-filtered) ----------------
% PRI sabit varsayımı -> eventler arası periyodiklik çok güçlü ipucu.
evTbl.PRIbin_local   = local_PRI_estimate(evTbl.tstartBin, evTbl.PWbin, opt.TolPW);
evTbl.PRIscore_local = local_periodicity_score(evTbl.tstartBin, evTbl.PRIbin_local);

%% ---------------- 6) Cluster events with PRI + periodicity (FSK tolerant to Fc) ----------------
[clusterId, clusters] = cluster_events_PRI_pw(evTbl, opt);

%% ---------------- 7) Cluster descriptive words + label mapping ----------------
[clusters, evTbl] = label_clusters_and_words(clusters, evTbl, opt);

%% ---------------- 8) PRI final per cluster (single PRI per class) ----------------
clusters = finalize_cluster_PRI(clusters, evTbl);

%% ---------------- 9) Pack output ----------------
out = struct();
out.IB = IB;
out.IC = IC;

out.events = evTbl;

out.eventClusterId = clusterId;
out.clusters = clusters;

out.clusterWord  = {clusters.word};
out.clusterLabel = {clusters.label};
out.clusterPRIbin = [clusters.PRIbin];
out.clusterPRIt   = [clusters.PRIt];

% quick print
fprintf('\n=== Clusters ===\n');
for k=1:numel(clusters)
    fprintf('C%02d | n=%d | PRI(bin)=%.2f | label=%s | word=%s\n', ...
        k, clusters(k).nEvents, clusters(k).PRIbin, clusters(k).label, clusters(k).word);
end

end

%% ========================================================================
%  OTSU + POSTPROCESS
% ========================================================================
function [IB, IC] = otsu_threshold_and_post(ID_gray01, p)
T0 = graythresh(ID_gray01) * p.k;
IB = imbinarize(ID_gray01, T0);
IC = postprocess_IC(IB, p);
end

function IC = postprocess_IC(IB, p)
alpha = p.alpha;
beta1 = p.beta1;
beta2 = p.beta2;

% (1) time-direction closing (split killer)
if isfield(p,'closeT') && p.closeT > 0
    closeT = make_odd(round(p.closeT));
    IB = imclose(IB, strel('line', closeT, 0)); % yatay
end

% (2) CCA small component removal
CC = bwconncomp(IB, 8);
stats = regionprops(CC, 'Area');
areas = [stats.Area];
IC = IB;
smallIdx = find(areas < alpha);
for ii = smallIdx
    IC(CC.PixelIdxList{ii}) = 0;
end

% optional extra minArea
if isfield(p,'minArea') && p.minArea > 0
    CC2 = bwconncomp(IC, 8);
    st2 = regionprops(CC2,'Area');
    ar2 = [st2.Area];
    small2 = find(ar2 < p.minArea);
    for ii = small2
        IC(CC2.PixelIdxList{ii}) = 0;
    end
end

[F,~] = size(IC);

% (3) time mask V + gap fill
V = any(IC,1);
S = find(V==1);
if numel(S)>=2
    for i=1:numel(S)-1
        if S(i+1)-S(i) < beta1
            V(S(i):S(i+1)) = 1;
        end
    end
end

% (4) freq mask U + gap fill
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

% (5) apply masks
IC = IC & (U * V);
end

function v = make_odd(v)
v = max(3, v);
if mod(v,2)==0, v = v+1; end
end

%% ========================================================================
%  IC -> BLOBS
% ========================================================================
function blobs = extract_blobs_from_IC(IC)
CC = bwconncomp(IC, 8);
if CC.NumObjects==0
    blobs = struct([]);
    return;
end
rp = regionprops(CC, 'BoundingBox', 'Area', 'PixelIdxList');

blobs = struct([]);
for i=1:CC.NumObjects
    bb = rp(i).BoundingBox; % [x y w h] (x:col, y:row)
    tstart = floor(bb(1)) + 1;
    fstart = floor(bb(2)) + 1;
    tstop  = min(tstart + floor(bb(3)) - 1, size(IC,2));
    fstop  = min(fstart + floor(bb(4)) - 1, size(IC,1));
    blobs(i).tstart = tstart;
    blobs(i).tstop  = tstop;
    blobs(i).fstart = fstart;
    blobs(i).fstop  = fstop;
    blobs(i).area   = rp(i).Area;
    blobs(i).pix    = rp(i).PixelIdxList;
end

% sort by time
[~,idx] = sort([blobs.tstart]);
blobs = blobs(idx);
end

%% ========================================================================
%  BLOBS -> EVENTS (MERGE SPLITS)
% ========================================================================
function events = blobs_to_events_merge(blobs, mergeGapT, mergePwTol)
if isempty(blobs)
    events = struct([]);
    return;
end

% Initial: each blob as an event candidate
ev = struct([]);
for i=1:numel(blobs)
    ev(i).tstart = blobs(i).tstart;
    ev(i).tstop  = blobs(i).tstop;
    ev(i).fstart = blobs(i).fstart;
    ev(i).fstop  = blobs(i).fstop;
    ev(i).area   = blobs(i).area;
    ev(i).numSubBlobs = 1;
end

% Merge pass (time-ordered)
merged = true;
while merged
    merged = false;
    i=1;
    newEv = struct([]);
    c = 0;
    while i <= numel(ev)
        cur = ev(i);
        j = i+1;
        while j <= numel(ev)
            nxt = ev(j);

            gapT = nxt.tstart - cur.tstop;
            if gapT > mergeGapT
                break; % since sorted by tstart
            end

            PWcur = cur.tstop - cur.tstart + 1;
            PWnxt = nxt.tstop - nxt.tstart + 1;

            pwOK = abs(PWcur - PWnxt) <= mergePwTol;

            % time-overlap or close
            timeOK = (nxt.tstart <= cur.tstop) || (gapT <= mergeGapT);

            if timeOK && pwOK
                % merge
                cur.tstart = min(cur.tstart, nxt.tstart);
                cur.tstop  = max(cur.tstop,  nxt.tstop);
                cur.fstart = min(cur.fstart, nxt.fstart);
                cur.fstop  = max(cur.fstop,  nxt.fstop);
                cur.area   = cur.area + nxt.area;
                cur.numSubBlobs = cur.numSubBlobs + nxt.numSubBlobs;
                merged = true;

                % remove nxt by skipping it
                j = j + 1;
                % mark nxt as consumed
                ev(j-1).tstart = inf; %#ok<AGROW>
            else
                j = j + 1;
            end
        end

        if isfinite(cur.tstart)
            c = c + 1;
            newEv(c) = cur; %#ok<AGROW>
        end
        i = i + 1;
    end

    % cleanup + sort
    ev = newEv;
    [~,idx] = sort([ev.tstart]);
    ev = ev(idx);
end

events = ev;
end

%% ========================================================================
%  EVENTS -> TABLE (bin + physical)
% ========================================================================
function evTbl = events_to_table(events, tVec, fVec)
n = numel(events);

tstartBin = zeros(n,1);
tstopBin  = zeros(n,1);
fstartBin = zeros(n,1);
fstopBin  = zeros(n,1);
PWbin     = zeros(n,1);
BWbin     = zeros(n,1);
Fcbin     = zeros(n,1);
numSub    = zeros(n,1);
area      = zeros(n,1);

for i=1:n
    tstartBin(i) = events(i).tstart;
    tstopBin(i)  = events(i).tstop;
    fstartBin(i) = events(i).fstart;
    fstopBin(i)  = events(i).fstop;
    PWbin(i)     = tstopBin(i) - tstartBin(i) + 1;
    BWbin(i)     = fstopBin(i) - fstartBin(i) + 1;
    Fcbin(i)     = 0.5*(fstartBin(i) + fstopBin(i));
    numSub(i)    = events(i).numSubBlobs;
    area(i)      = events(i).area;
end

% Physical
tstart = tVec(max(1,min(numel(tVec), tstartBin)));
tstop  = tVec(max(1,min(numel(tVec), tstopBin)));
PWt    = tstop - tstart;

fstart = fVec(max(1,min(numel(fVec), fstartBin)));
fstop  = fVec(max(1,min(numel(fVec), fstopBin)));
BWf    = fstop - fstart;
Fcf    = 0.5*(fstart + fstop);

evTbl = table( ...
    (1:n).', tstartBin, tstopBin, fstartBin, fstopBin, ...
    PWbin, BWbin, Fcbin, numSub, area, ...
    tstart(:), tstop(:), PWt(:), fstart(:), fstop(:), BWf(:), Fcf(:), ...
    'VariableNames', {'id','tstartBin','tstopBin','fstartBin','fstopBin', ...
                      'PWbin','BWbin','Fcbin','numSubBlobs','area', ...
                      'tstart','tstop','PWt','fstart','fstop','BWf','Fcf'} );
end

%% ========================================================================
%  LOCAL PRI ESTIMATE + PERIODICITY SCORE
% ========================================================================
function PRI_local = local_PRI_estimate(tstartBin, PWbin, tolPW)
n = numel(tstartBin);
PRI_local = nan(n,1);

for i=1:n
    % candidates with similar PW
    idx = find(abs(PWbin - PWbin(i)) <= tolPW);
    idx(idx==i) = [];
    if numel(idx) < 2
        continue;
    end
    dt = abs(tstartBin(idx) - tstartBin(i));
    dt = dt(dt>0);
    if isempty(dt), continue; end

    % robust "mode-like" using histogram
    PRI_local(i) = robust_mode_bin(dt);
end
end

function s = local_periodicity_score(tstartBin, PRIbin)
% Score: how well tstart aligns to a grid with step PRIbin
% 0..1 (1 is perfect)
n = numel(tstartBin);
s = zeros(n,1);

for i=1:n
    p = PRIbin(i);
    if ~isfinite(p) || p <= 1
        s(i) = 0;
        continue;
    end
    % nearest grid offset: minimize residue
    % Using mod with multiple offsets approximated by median residue
    r = mod(tstartBin(i), p);
    r = min(r, p-r); % distance to nearest multiple
    s(i) = max(0, 1 - (r / max(p,1)));
end
end

function m = robust_mode_bin(x)
x = x(:);
x = x(isfinite(x) & x>0);
if isempty(x), m = nan; return; end
% bin width: 1 bin
xmin = min(x); xmax = max(x);
edges = (xmin-0.5):(1):(xmax+0.5);
if numel(edges) < 3
    m = median(x);
    return;
end
h = histcounts(x, edges);
[~,k] = max(h);
centers = edges(1:end-1) + 0.5;
m = centers(k);
end

%% ========================================================================
%  CLUSTERING (PRI + PW heavy, Fc relaxed for FSK)
% ========================================================================
function [clusterId, clusters] = cluster_events_PRI_pw(evTbl, opt)
n = height(evTbl);

% Initialize: unassigned
clusterId = zeros(n,1);

% Precompute a "good PRI" flag
goodPRI = isfinite(evTbl.PRIbin_local) & evTbl.PRIbin_local > 1 & (evTbl.PRIscore_local >= opt.PerScoreMin);

% Sort events by time
[~,ord] = sort(evTbl.tstartBin);
invOrd = zeros(n,1); invOrd(ord) = 1:n;

cid = 0;

for kk = 1:n
    i = ord(kk);
    if clusterId(i) ~= 0
        continue;
    end

    cid = cid + 1;
    seed = i;

    % Seed stats
    members = seed;
    clusterId(seed) = cid;

    % grow cluster by iterative inclusion
    changed = true;
    while changed
        changed = false;

        % current cluster medians
        PWm  = median(evTbl.PWbin(members));
        BWm  = median(evTbl.BWbin(members));
        Fcm  = median(evTbl.Fcbin(members));
        PRIm = median(evTbl.PRIbin_local(members), 'omitnan');

        % Determine if FSK-like candidate: Fc can hop but PRI+PW stable
        FcStd = std(evTbl.Fcbin(members));
        PWStd = std(evTbl.PWbin(members));
        PRIStd = std(evTbl.PRIbin_local(members), 'omitnan');

        isFSKcand = (FcStd >= opt.FcHopStd) && (PWStd <= opt.TolPW/2) && (PRIStd <= opt.TolPRI/2);

        for j = 1:n
            if clusterId(j) ~= 0
                continue;
            end

            % Must be close in PW
            if abs(evTbl.PWbin(j) - PWm) > opt.TolPW
                continue;
            end

            % Must have compatible PRI (if available)
            if isfinite(PRIm) && isfinite(evTbl.PRIbin_local(j))
                if abs(evTbl.PRIbin_local(j) - PRIm) > opt.TolPRI
                    continue;
                end
                % periodicity score also should be decent
                if evTbl.PRIscore_local(j) < opt.PerScoreMin
                    continue;
                end
            else
                % If PRI is not known, allow only if it is near other members in time differences
                if ~weak_time_support(evTbl.tstartBin(members), evTbl.tstartBin(j), opt.TolPRI)
                    continue;
                end
            end

            % BW compatibility (loose)
            if abs(evTbl.BWbin(j) - BWm) > opt.TolBW
                % allow phase-coded family to have more BW variation
                if median(evTbl.numSubBlobs(members)) < 2
                    continue;
                end
            end

            % Fc compatibility:
            if ~isFSKcand
                if abs(evTbl.Fcbin(j) - Fcm) > opt.TolFc
                    % but if both are goodPRI, PRI dominates more than Fc (allow)
                    if ~(goodPRI(j) && any(goodPRI(members)))
                        continue;
                    end
                end
            end

            % add
            members(end+1) = j; %#ok<AGROW>
            clusterId(j) = cid;
            changed = true;
        end
    end
end

% Build cluster structs
clusters = struct([]);
for c = 1:cid
    idx = find(clusterId==c);
    clusters(c).id = c;
    clusters(c).members = idx(:).';
    clusters(c).nEvents = numel(idx);

    clusters(c).PWm  = median(evTbl.PWbin(idx));
    clusters(c).BWm  = median(evTbl.BWbin(idx));
    clusters(c).Fcm  = median(evTbl.Fcbin(idx));
    clusters(c).PRIbin_raw = median(evTbl.PRIbin_local(idx), 'omitnan');

    clusters(c).FcStd  = std(evTbl.Fcbin(idx));
    clusters(c).PWStd  = std(evTbl.PWbin(idx));
    clusters(c).PRIStd = std(evTbl.PRIbin_local(idx), 'omitnan');

    clusters(c).numSubMed = median(evTbl.numSubBlobs(idx));
    clusters(c).perScoreMed = median(evTbl.PRIscore_local(idx), 'omitnan');

    clusters(c).word  = "";
    clusters(c).label = "";
    clusters(c).PRIbin = NaN;
    clusters(c).PRIt   = NaN;
end
end

function ok = weak_time_support(tMembers, tCand, tolPRI)
% If candidate has dt close to some member dt, accept weakly
dt = abs(tMembers - tCand);
dt = dt(dt>0);
if isempty(dt), ok=false; return; end
m = robust_mode_bin(dt);
ok = isfinite(m) && (m <= 1e9) && (m >= 2) && (abs(dt(1) - m) <= tolPRI || abs(m - median(dt)) <= tolPRI);
end

%% ========================================================================
%  CLUSTER WORDS + LABEL MAPPING
% ========================================================================
function [clusters, evTbl] = label_clusters_and_words(clusters, evTbl, opt)

for c = 1:numel(clusters)
    idx = clusters(c).members;

    PWm  = clusters(c).PWm;
    BWm  = clusters(c).BWm;
    FcStd = clusters(c).FcStd;
    subM  = clusters(c).numSubMed;

    PRIm = clusters(c).PRIbin_raw;
    perM = clusters(c).perScoreMed;

    % --- Descriptive word (rule-based) ---
    isPhaseCodedLike = (subM >= 2); % lots of sub-blobs after merge indicates chippy/splitty
    isWideBW         = (BWm >= 60); % tune per your spec
    isFSKLike        = (FcStd >= opt.FcHopStd) && (clusters(c).PWStd <= opt.TolPW/2) && (clusters(c).PRIStd <= opt.TolPRI/2);

    if isFSKLike
        word = "MULTITONE_FSKLIKE_PRI_LOCKED";
    elseif isPhaseCodedLike
        word = "PHASE_CODED_MULTI_BLOB_PRI_LOCKED";
    elseif isWideBW
        word = "CHIRP_WIDEBW_PRI_LOCKED";
    else
        word = "NARROW_OR_SIMPLE_PRI_LOCKED";
    end

    % --- Label mapping ---
    % Note: LFM vs NLFM separation from IC alone is hard; we add simple curvature proxy
    label = "unknown";

    if isFSKLike
        label = "FSK";
    elseif isPhaseCodedLike
        % Frank vs P_All heuristic: if PW/BW suggests N^2 grid-ish -> Frank else P_All
        if looks_like_frank(evTbl(idx,:))
            label = "Frank";
        else
            label = "P_All";
        end
    else
        % chirp-like vs barker-like: simple ridge linearity
        if isWideBW
            % decide LFM vs NLFM by ridge linearity score
            if ridge_is_linear(evTbl(idx,:))
                label = "LFM";
            else
                label = "NLFM";
            end
        else
            % default: Barker if short PW and moderate BW
            if PWm <= 30 && BWm >= 20
                label = "Barker";
            else
                label = "LFM"; % fallback
            end
        end
    end

    clusters(c).word  = word;
    clusters(c).label = label;

    % write per-event label/word
    evTbl.clusterWord(idx)  = repmat(string(word), numel(idx), 1);
    evTbl.clusterLabel(idx) = repmat(string(label), numel(idx), 1);

    % store diagnostic
    clusters(c).PRIbin_raw = PRIm;
    clusters(c).perScoreMed = perM;
end

% Ensure columns exist
if ~ismember('clusterWord', evTbl.Properties.VariableNames)
    evTbl.clusterWord = repmat("", height(evTbl),1);
end
if ~ismember('clusterLabel', evTbl.Properties.VariableNames)
    evTbl.clusterLabel = repmat("", height(evTbl),1);
end

end

function tf = looks_like_frank(evSub)
% Frank heuristic: often many chips -> tends to larger numSubBlobs and
% BW/PW ratio characteristic. This is a weak heuristic; you can refine later.
subM = median(evSub.numSubBlobs);
PWm  = median(evSub.PWbin);
BWm  = median(evSub.BWbin);
ratio = BWm / max(PWm,1);

tf = (subM >= 3) && (ratio >= 1.0);
end

function tf = ridge_is_linear(evSub)
% Very lightweight proxy: if BW is stable and numSubBlobs small, assume LFM.
% (You can replace this with true ridge extraction later.)
BWstd = std(evSub.BWbin);
subM = median(evSub.numSubBlobs);
tf = (BWstd <= 10) && (subM <= 1);
end

%% ========================================================================
%  FINAL PRI per cluster (single PRI assumption)
% ========================================================================
function clusters = finalize_cluster_PRI(clusters, evTbl)

for c = 1:numel(clusters)
    idx = clusters(c).members;

    tstarts = sort(evTbl.tstartBin(idx));
    if numel(tstarts) >= 2
        dt = diff(tstarts);
        dt = dt(dt>0);

        PRIbin = robust_mode_bin(dt);
        if ~isfinite(PRIbin)
            PRIbin = median(dt);
        end
    else
        PRIbin = NaN;
    end

    clusters(c).PRIbin = PRIbin;

    % Physical PRI if tVec exists in evTbl (we have tstart seconds)
    % Use median(dtSeconds) from tstart column:
    tstarts_s = sort(evTbl.tstart(idx));
    if numel(tstarts_s) >= 2
        dts = diff(tstarts_s);
        dts = dts(dts>0);
        PRIt = median(dts);
    else
        PRIt = NaN;
    end
    clusters(c).PRIt = PRIt;
end

end
