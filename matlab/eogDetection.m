% Computes EOG saccade rapidity and slowniless components
% PreCD: "signal" is already the desired EOG derivation (better sort E2-E1 horizontal), 
%        which has been already Notch and ECG-artifact filtered
function [rapidity, slowniness] = eogDetection(signal, sr, relEOG, intTime)

%% Preprocessing of the SEM component
disp('Preprocessing of the SEM component...');
semNorm = zeros(size(signal));
%fgain = 1;
fhp = 0.15;
flp = 0.3;
[~, bpEOG] = freqSEManalysis(signal, sr, fhp, flp, 0.4, 10); % 08/03/2016 GOOD ONE SO FAR!!!!

fprintf(1, '\nStatistics freqSEManalysis\n');
fprintf(1, '\nMean: %.2f', mean(bpEOG));
fprintf(1, '\nMedian: %.2f', median(bpEOG));
fprintf(1, '\nMin-Max: [%.2f - %.2f]', min(bpEOG), max(bpEOG));
fprintf(1, '\nRatio zeros: %.2f', sum(bpEOG == 0)/length(bpEOG));
fprintf(1, '\nRatio 100s: %.2f\n', sum(bpEOG == 100)/length(bpEOG));

%% Saccade detection
%% Signal differentiation on a certain T
T = intTime;
sT = round(T * sr);
signalD = [signal(sT+1:end); signal(end)*ones(sT, 1)];
fsignal = signalD - signal;

%% Computation of Vmin (baseline)
disp('EOG saccade detection: detecting Vmin...');
wsize = 10; %seconds
step = 1; % In seconds (before 10)
percentile = 20; % To determine baseline amplitude in the window (in percentage)
detectFactor = 10; % default 10!!!
steps = round(step * sr); % algorithm step in samples
wsizes = round(wsize * sr); % window size in samples
numWindows = floor((length(fsignal) - wsizes)/steps);

%progressbar = waitbar(0, sprintf('Detecting Vmin...'));
%movegui(progressbar, 'center');
baseline = zeros(size(fsignal));
fsignalRect = abs(fsignal);
for k = 1:numWindows
    idxw = max(1, (k -1)*steps + 1 - wsizes/2):min(length(fsignal), (k-1)*steps + wsizes/2);
    window = fsignalRect(idxw);

    baseline(idxw) = detectFactor * prctile(window, percentile);
    
    %if mod(k,floor(numWindows/10))==0
    %    waitbar((k / numWindows), progressbar);
    %end
end
%close(progressbar);
disp('done!');

%% Compute peak regions
disp('EOG saccade detection: computing peak-regions...');

thresholdPeak = 1;

saccAreas = zeros(size(fsignal));
%peakRegions = zeros(size(fsignal), 'int8');

for k = 1:2

    % It has to be done like this to prevent detection areas where we are
    % constantly changing from + threshold to - threshold.
    if (k == 1)
        peakRegionsTemp = fsignal > baseline * thresholdPeak;
    else
        peakRegionsTemp = fsignal < -baseline * thresholdPeak;
    end
       
    inisP = find(diff(peakRegionsTemp) == 1) + 1;
    endsP = find(diff(peakRegionsTemp) == -1);

    if peakRegionsTemp(1)
        inisP = [1; inisP];
    end
    if peakRegionsTemp(end)
        endsP = [endsP; length(peakRegionsTemp)];
    end

    if length(inisP) > length(endsP)
        inisP = inisP(1:end-1);
    end
    if length(endsP) > length(inisP)
        endsP = endsP(2:end);
    end
    
    if ne(length(inisP), length(endsP))
        disp('Error: different number of start/end points');
        return;
    end
    
    % Reject peaks due to electrode-artefact or EMG by some threshold and
    % duration. We avoid also false saccades due to sampling rate aliasing
    
    % Starting from the points that have crossed the threshold, we its
    % duration is at least minDur
    minDur = 50/1000; %ms ORIGINAL!!: 256Hz=>0.004 s per sample, thus 50ms=0.05s=> at least 12.5 samples
    minDur2 = 50/1000;
    
    % When detected a valid saccade over the MinDur threshold, it may be
    % accepted to be increase its length if its velocity does not decay
    % under propSpeed of its maxSpeed and it keeps over a reduced proportion 
    % of the the baseline threshold determined by allowFactor (it helps to admit brief decays 
    %of velocity during the saccade due to noise, but that still can be considered within the same saccade event) 
    propSpeed = 0.5; %proportion to the orignal speed
    allowFactor = 0.5; 
               
    %progressbar = waitbar(0, sprintf('Analyzing saccades...'));
    %movegui(progressbar, 'center');
    
    nextIni = 1;
    for k1 = 1:length(inisP)
        
        if (inisP(k1) < nextIni)
            continue;
        end
        
        saccStart = inisP(k1);
        saccEnd = endsP(k1);
        
        % Reject those below minDur
        if ((saccEnd - saccStart)/sr < minDur)
            continue;
        end
        
        % Given the duration is acceptable, we start checking there is no
        % waxing and waving in the EOG (fine-tunning, not really crucial as with the article's configuration)
        
        % Find the highest peak to compute maximum velocity
        if (k == 1)
            [maxVel, indxMaxVel] = max(fsignal(saccStart:saccEnd));
        elseif (k == 2)
            [maxVel, indxMaxVel] = min(fsignal(saccStart:saccEnd));
        end
        refVel = propSpeed * (maxVel/(T*1000)); % uV/ms
        
        % Now, from the point of maximum speed, we look forward and
        % backwards in the original signal to stablish the begining and the end of the saccade
        saccStart = inisP(k1) + indxMaxVel - 1;
        saccEnd = min(saccStart + sT, length(signal)); % Since the filtered signal is calculated using differences every sT samples
        saccStartTmp = saccStart;
        saccEndTmp = saccEnd;
        if (k == 1)
            % Forward
            if saccEndTmp < length(signal)
                while ((signal(saccEndTmp + 1) - signal(saccStart))/(1000*(saccEndTmp+1-saccStart)/sr) > refVel) && (fsignal(saccEndTmp+1) > allowFactor*baseline(saccEndTmp+1))
                    saccEndTmp = saccEndTmp + 1;
                    if (saccEndTmp == length(signal))
                        break;
                    end
                end
            end
            % Backwards
            if saccStartTmp > 1
                while ((signal(saccEnd) - signal(saccStartTmp-1))/(1000*(saccEnd-saccStartTmp-1)/sr) > refVel) && (fsignal(saccStartTmp-1) > allowFactor*baseline(saccStartTmp-1))
                    saccStartTmp = saccStartTmp - 1;
                    if (saccStartTmp == 1)
                        break;
                    end
                end
            end
        elseif (k == 2)
            % Forward
            if saccEndTmp < length(signal)
                while ((signal(saccEndTmp+1) - signal(saccStart))/(1000*(saccEndTmp+1-saccStart)/sr) < refVel) && (fsignal(saccEndTmp+1) < -allowFactor*baseline(saccEndTmp+1))
                    saccEndTmp = saccEndTmp + 1;
                    if (saccEndTmp == length(signal))
                        break;
                    end
                end
            end
            % Backwards
            if saccStartTmp > 1
                while ((signal(saccEnd) - signal(saccStartTmp-1))/(1000*(saccEnd-saccStartTmp-1)/sr) < refVel) && (fsignal(saccStartTmp-1) < -allowFactor*baseline(saccStartTmp-1))
                    saccStartTmp = saccStartTmp - 1;
                    if (saccStartTmp == 1)
                        break;
                    end
                end
            end
        end
        saccStart = saccStartTmp;
        saccEnd = saccEndTmp;
        
        % Recheck again for minimum duration
        if ((saccEnd - saccStart)/sr < minDur2)
            continue;
        end
        
        % Annotation and quantification of the resulting saccade
        delay = 0;
        saccStartD = saccStart + delay;
        saccEndD = saccEnd + delay;

        % Calculates corresponding area (real integration) in uV*s
        normFact = mean(baseline(saccStart:saccEnd));
        % Note: The only reason normFact can be zero is because baseline is
        %       zero. Baseline of zero can only be caused by zero signal,
        %       thus an artifact and therefore can (and should) be skipped.
        if (normFact > 0) && not(any(relEOG(saccStart:saccEnd) == 0)) 
            
            saccAreas(saccStartD:saccEndD) = sum(fsignalRect(saccStart:saccEnd))/normFact;
                                        
            saccInfluence = 4; %in seconds
            offset = round(saccInfluence*sr);
            
            % Filter saccades from SEM area by replacing with zeros
            centerSacPoint = min(saccStart + round(length(saccStart:saccEnd)/2), length(bpEOG));
            bpEOG(max(1, centerSacPoint-offset):min(centerSacPoint+offset, length(bpEOG))) = 0;

            %peakRegions(saccStartD:saccEndD) = 1;
        end
        
        % Next time we can start examining from here
        nextIni = saccEnd;
        
        % Update progress bar
        %if mod(k1,floor(length(inisP)/10))==0
        %    waitbar((k1 / length(inisP)), progressbar);
        %end
        
    end
    
    %close(progressbar);
    
end

% Remove false positives by reliability analysis
saccAreas(relEOG == 0) = 0;
%peakRegions(relEOG == 0) = 0;
bpEOG(relEOG == 0) = 0;

%saccades = peakRegions;

disp('done!');

rapidity = zeros(1, length(saccAreas));
slowniness = zeros(1, length(saccAreas));

%progressbar = waitbar(0, sprintf('Normalizing REM/SEM...'));
%movegui(progressbar, 'center');
wsize = 30; %seconds
step = 30; % In seconds (before 10)
steps = round(step * sr); % algorithm step in samples
wsizes = round(wsize * sr); % window size in samples
numWindows = floor((length(fsignal) - wsizes)/steps);
%semNormFact = mean(abs(bpEOG));
for k = 1:numWindows
    idxw = max(1, (k -1)*steps + 1):min(length(semNorm), (k-1)*steps + wsizes);

    % Actual normalization
    rapidity(idxw) = sum(saccAreas(idxw))/sr;
    slowniness(idxw) = sum(bpEOG(idxw))/sr;
        
    %if mod(k,floor(numWindows/10))==0
    %    waitbar((k / numWindows), progressbar);
    %end
end
%close(progressbar);
