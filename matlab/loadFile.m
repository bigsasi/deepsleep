%LOADFILE load data from edf file, returning the all the data contained
% in the file

%% The code below is based on the methods described in the following reference(s):
% 
% [1] - I. Fernández-Varela, D. Alvarez-Estevez, E. Hernández-Pereira, V. Moret-Bonillo, 
% "A simple and robust method for the automatic scoring of EEG arousals in
% polysomnographic recordings", Computers in Biology and Medicine, vol. 87, pp. 77-86, 2017 
%
% Copyright (C) 2017 Isaac Fernández-Varela
% Copyright (C) 2017 Diego Alvarez-Estevez

%% This program is free software: you can redistribute it and/or modify
%% it under the terms of the GNU General Public License as published by
%% the Free Software Foundation, either version 3 of the License, or
%% (at your option) any later version.

%% This program is distributed in the hope that it will be useful,
%% but WITHOUT ANY WARRANTY; without even the implied warranty of
%% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%% GNU General Public License for more details.

%% You should have received a copy of the GNU General Public License
%% along with this program.  If not, see <http://www.gnu.org/licenses/>.

function [edfFile] = loadFile(filename, confXML)
    hdr = EDFreadHeader(filename);
    if isempty(hdr)
        error('Error reading input EDF header. Bad file name?');
    end
    
    edfFile.header = hdr;

    indxMontage = -1;
    for k = 1:length(confXML.montage)
       if matchMontage(confXML.montage(k), hdr)
           indxMontage = k;
           break;
       end
    end
    
    if (indxMontage == -1)
        error('Not able to found a valid montage');
    end

    signals = {'eeg1', 'eeg2', 'emg', 'ecg', 'eog1', 'eog2'};
    
    for i=1:length(signals)
        signalName = signals{i};
        [raw, channel, label] = readSignal(filename, confXML, indxMontage, signalName);
        edfFile.signal{i}.label = label;
        edfFile.signal{i}.rate = hdr.signals_info(channel).sample_rate;
        edfFile.signal{i}.raw = raw;
        edfFile.(signalName) = i;
    end
        
    edfFile.header.num_signals = i;
end


function [raw, channel, label] = readSignal(inputEDFname, confXML, indxMontage, signalName)
    raw = [];
    
    channelMapping = confXML.montage(indxMontage).channelmapping;
    
    for k = 1:length(channelMapping.(signalName).channel)
        if (strcmpi(channelMapping.(signalName).channel(k).ATTRIBUTE.sign, 'plus'))
            channel = channelMapping.(signalName).channel(k).ATTRIBUTE.idx;
            if (k == 1)
                raw = EDFreadSignal(inputEDFname, channel, [], []);
            else
                raw = raw + EDFreadSignal(inputEDFname, channel, [], []);
            end
        elseif (strcmpi(channelMapping.(signalName).channel(k).ATTRIBUTE.sign, 'minus'))
            channel = channelMapping.(signalName).channel(k).ATTRIBUTE.idx;
            if (k == 1)
                raw = -EDFreadSignal(inputEDFname, channel, [], []);
            else
                raw = raw - EDFreadSignal(inputEDFname, channel, [], []);
            end
        else
            error(['Error when parsing channel mapping for ' signalName]);
        end
    end
    
    label = channelMapping.(signalName).ATTRIBUTE.label';
end

function result = matchMontage(xmlMontageId, hdr)
    result = 1;
    
    channelMapping = xmlMontageId.channelmapping;
    
    signalsToCheck = {'eeg1', 'eeg2', 'emg', 'eog1', 'eog2', 'airflow', 'abdoRes', 'thorRes', 'saturation', 'position'};
    
    for i = 1:length(signalsToCheck)
        signal = signalsToCheck{i};
        for k = 1:length(channelMapping.(signal).channel)
            channel = xmlMontageId.channelmapping.(signal).channel(k);
            label = channel.ATTRIBUTE.label;
            idxChannel = channel.ATTRIBUTE.idx;
            result = result && (findIndexChannelByLabel(hdr, label) == idxChannel);
        end
    end
end

function [indx] = findIndexChannelByLabel(hdr, labelStr)
    indx = -1;

    for k = 1:length(hdr.signals_info)
        if (strncmpi(labelStr, hdr.signals_info(k).label, max(length(strtrim(sprintf('%s', hdr.signals_info(k).label))), length(labelStr))))
            indx = k;
            return;
        end
    end
end
