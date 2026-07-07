function show_dataset_examples()
% Veri setinden 3x3 ornek gorsel uretir (sunum icin).
%
% Satir 1: Gurultulu spektrogram
% Satir 2: Temiz (referans) spektrogram
% Satir 3: |Gurultulu - Temiz| farki
%
% Secim kriteri:
%   - SNR degerleri {0, 10, 20} dB
%   - numTypes = 3 (ayni yamada 3 sinif aktif)

clc; close all;

% ===================== AYARLAR =====================
% Tum yollar bu betigin konumundan (repo koku) turetilir; makineye ozel
% mutlak yol yoktur.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
dataRoot    = fullfile(projectRoot, 'dataset', 'dataset_multiframe_patches');

labelsPath = fullfile(dataRoot, 'labels.csv');
noisyDir   = fullfile(dataRoot, 'specNoisyGray');
cleanDir   = fullfile(dataRoot, 'specCleanGray');

targetSNRs = [0 10 20];
K = numel(targetSNRs);     % = 3
saveFig = true;
outPng  = fullfile(dataRoot, 'example_3x3_SNR_0_10_20_numTypes3.png');

showLabelsInTitle = true;
maxLabelChars     = 42;

% ===================== LOAD LABELS =====================
assert(exist(labelsPath,'file')==2, "labels.csv bulunamadı");
T = readtable(labelsPath);

names = string(T{:,1});

% --- class kolonlarını otomatik bul (binary olanlar) ---
varNames = string(T.Properties.VariableNames);
classVars = varNames(2:end);

isBinaryCol = false(size(classVars));
for i=1:numel(classVars)
    v = T.(classVars(i));
    if isnumeric(v)
        u = unique(v(~isnan(v)));
        isBinaryCol(i) = all(ismember(u,[0 1]));
    end
end
classVars = classVars(isBinaryCol);

C = numel(classVars);

% --- numTypes = aktif class sayısı ---
Y = zeros(height(T),1);
for i=1:C
    Y = Y + T.(classVars(i));
end

% ===================== ÖRNEK SEÇ =====================
selNames = strings(1,K);

for k = 1:K
    snr = targetSNRs(k);

    idx = find( ...
        T.SNR_dB == snr & ...
        Y == 3 ...
    );

    assert(~isempty(idx), ...
        "SNR=%d ve numTypes=3 için örnek bulunamadı", snr);

    pick = idx(randi(numel(idx)));
    selNames(k) = names(pick);
end

% ===================== FIGURE =====================
hFig = figure('Name','Dataset Examples (SNR=0/10/20, numTypes=3)', ...
              'Position',[100 80 1200 900]);
tl = tiledlayout(3, K, 'Padding','compact', 'TileSpacing','compact');

for c = 1:K
    fnameNoExt = selNames(c);
    fnamePng   = fnameNoExt + ".png";

    In = im2single(imread(fullfile(noisyDir, fnamePng)));
    Ic = im2single(imread(fullfile(cleanDir, fnamePng)));

    if ndims(In)==3, In = rgb2gray(In); end
    if ndims(Ic)==3, Ic = rgb2gray(Ic); end

    Id = abs(In - Ic);

    % --- label string ---
    lblStr = "";
    r = find(names == fnameNoExt, 1);
    if showLabelsInTitle && ~isempty(r)
        onIdx = [];
        for i=1:C
            if T.(classVars(i))(r) == 1
                onIdx(end+1) = i; %#ok<AGROW>
            end
        end
        lblStr = strjoin(classVars(onIdx), "+");
        if strlength(lblStr) > maxLabelChars
            lblStr = extractBefore(lblStr, maxLabelChars) + "...";
        end
    end

    snrVal = T.SNR_dB(r);

    % --- Row 1: Noisy ---
    nexttile(tl, c);
    imshow(In, []);
    title(sprintf("Gürültülü | SNR=%d dB\n%s", snrVal, lblStr), ...
        'Interpreter','none');

    % --- Row 2: Clean ---
    nexttile(tl, K + c);
    imshow(Ic, []);
    title("Temiz", 'Interpreter','none');

    % --- Row 3: Diff ---
    nexttile(tl, 2*K + c);
    imshow(Id, []);
    title("|Gürültülü − Temiz|", 'Interpreter','none');
end

title(tl, ...
    "Üç Sinyal Türlü Çoklu Örnek Veri — SNR = 0 / 10 / 20 dB", ...
    'FontWeight','bold', 'Interpreter','none');

% ===================== SAVE =====================
if saveFig
    exportgraphics(hFig, outPng, 'Resolution', 200);
    fprintf("Gorsel kaydedildi: %s\n", outPng);
end

end
