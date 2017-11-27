function relEOG = eogReliabilityCheck(eog, sr)

%relEOG = ones(size(eog));

% Detection of sensor saturation intervals
satZones = int8(diff(eog) == 0);
realSaturations = zeros(size(eog), 'int8');

min_sat_dur = 0.5; %In seconds: minimun duration of a zero differentiated signal to be actually considered a sensor saturation artefact
min_sat_dur_samp = min_sat_dur * sr;

inisP = find(diff(satZones) == 1) + 1;
endsP = find(diff(satZones) == -1);

if satZones(1)
    inisP = [1; inisP];
end
if satZones(end)
    endsP = [endsP; length(satZones)];
end

if length(inisP) > length(endsP)
    inisP = inisP(1:end-1);
end
if length(endsP) > length(inisP)
    endsP = endsP(2:end);
end

for k = 1:length(inisP)
    if (length(inisP(k):endsP(k)) > min_sat_dur_samp)
        realSaturations(inisP(k):endsP(k)) = 1;
    end
end

union_time = 10; % In seconds: minimum time between two real saturation intevals, to be considered a reliable signal
union_samp = union_time * sr;

inisP = find(diff(realSaturations) == 1) + 1;
endsP = find(diff(realSaturations) == -1);

if realSaturations(1)
    inisP = [1; inisP];
end
if realSaturations(end)
    endsP = [endsP; length(realSaturations)];
end

if length(inisP) > length(endsP)
    inisP = inisP(1:end-1);
end
if length(endsP) > length(inisP)
    endsP = endsP(2:end);
end

% Checking intervals between two saturation intervals
for k = 2:length(inisP)
    if ((inisP(k)-endsP(k-1)) < union_samp)
        realSaturations(endsP(k-1):inisP(k)) = 1;
    end
end

offset = 10; % In seconds: time added around a real saturation interval in which the signal is still considered unreliable
offset_samp = offset * sr;

inisP = find(diff(realSaturations) == 1) + 1;
endsP = find(diff(realSaturations) == -1);

if realSaturations(1)
    inisP = [1; inisP];
end
if realSaturations(end)
    endsP = [endsP; length(realSaturations)];
end

if length(inisP) > length(endsP)
    inisP = inisP(1:end-1);
end
if length(endsP) > length(inisP)
    endsP = endsP(2:end);
end

% Extend resulting saturation intervals by some offset
for k = 1:length(inisP)
    realSaturations(endsP(k):min(length(realSaturations), endsP(k)+offset_samp)) = 1;
    realSaturations(max(1, inisP(k)-offset_samp):inisP(k)) = 1;
end
    
relEOG = not(realSaturations);




