originPath = '/home/sasi/edf/SHHS-250/';
destPath = '/home/sasi/edf/mat/';

files = cell(0, 0);

tmp = dir([originPath '*.edf']);
for i = 1:length(tmp)
    files{end + 1}.name = [originPath tmp(i).name];
end

shortFileName = cell(length(files), 1);

configurationMontage = xml_parseany('conf.xml');

for i = 1:length(files)
    file = files{i}.name;
    
    shortFileName{i} = file(end - 9:end - 4);

    fprintf('File: %s\n', shortFileName{i});

    edfFile = loadFile(file, configurationMontage);
        
    signal = edfFile.signal{edfFile.eog1};
    reliability = eogReliabilityCheck(signal.raw, signal.rate);
    [rap1, slow1] = eogDetection(signal.raw, signal.rate, reliability, 0.05);
    edfFile.signal{end + 1}.raw = rap1;
    edfFile.signal{end}.rate = signal.rate;
    edfFile.rap1 = length(edfFile.signal);
    
    edfFile.signal{end + 1}.raw = slow1;
    edfFile.signal{end} = signal.rate;
    edfFile.slow1 = length(edfFile.signal);
    
%     signal = edfFile.signal{edfFile.eog2};
%     reliability = eogReliabilityCheck(signal.raw, signal.rate);
%     [rap2, slow2] = eogDetection(signal.raw, signal.rate, reliability, 0.05);
    
    save([destPath shortFileName{i} '.mat'], 'edfFile');
end

