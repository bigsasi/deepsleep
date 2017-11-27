function [signalFreqF, signalFreqP] = freqSEManalysis(eogsignal, sr, hpFreq, lpFreq, hpFreqC, lpFreqC)

% Check for valid settings
if ((hpFreq <= 0) || (hpFreq > sr/2))
    disp('Error: High-pass cut-off does not match Nyquist limit');
    return;
elseif ((lpFreq <= 0) || (lpFreq > sr/2))
    disp('Error: Low-pass cut-off does not match Nyquist limit');
    return;
end

signalFreqF = zeros(size(eogsignal));
signalFreqP = zeros(size(eogsignal));

%progressbar = waitbar(0, sprintf('Frequency analysis...'));
%movegui(progressbar, 'center');
wsize = 30; %seconds
step = 30; % In seconds (before 10)
steps = round(step * sr); % algorithm step in samples
wsizes = round(wsize * sr); % window size in samples
numWindows = floor((length(eogsignal) - wsizes)/steps);
for k = 1:numWindows
    idxw = max(1, (k -1)*steps + 1):min(length(eogsignal), (k-1)*steps + wsizes);
    %idxw = max(1, (k -1)*steps + 1 -wsizes/2):min(length(signal), (k-1)*steps + wsizes/2);

    % Actual analysis
    [Pxx, F] = periodogram(eogsignal(idxw), rectwin(length(idxw)), length(idxw), sr);
    
    pband = sum(Pxx(F >= hpFreq & F <= lpFreq));
    pbandC = sum(Pxx(F >= hpFreqC & F <= lpFreqC) .* F(F >= hpFreqC & F <= lpFreqC));
        
    ptot = sum(Pxx);
     
    if ne(ptot, 0)
       signalFreqF(idxw) = 100*(pband/ptot);
    elseif ne(pband, 0)
       signalFreqF(idxw) = 100;
    end
    % if both are zero, result will be zero (because of initialization)
  
    if ne(pbandC, 0)
       signalFreqP(idxw) = pband/pbandC; % Note that 99.99% of the time we are in this case (checked)
    elseif ne(pband, 0)
        % If numerator different from zero and denominator zero, then this
        % is interpreted as a non-reliable value. This is very exceptional
        % case. We solve it by assigning the mean value of SEM feature, but
        % probably is so minor that it does not affect the validation
        % results
        signalFreqP(idxw) = mean(signalFreqP(ne(signalFreqP, 0)));
    end
    % if both are zero, result will be zero (because of
    % initialization). Note, numerator zero necessarily means "absence of SEM feature"
    % with independence of denominator
    
    %if mod(k,floor(numWindows/10))==0
    %    waitbar((k / numWindows), progressbar);
    %end
end
%close(progressbar);