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

function statusok = annotations2EDFplus(annotations, patId, recId, startdate, starttime, filename)

% Compute how many characters we will need to store
charsToWrite = 0;

% Note: In this implementation we assume only one data block, thus only
%       one time-keeping TAL is necessary. This tal is: +0'20''20''0'
data = sprintf('%s%s', '+0', [20,20,0]);
charsToWrite = charsToWrite + 5;

% Each annotation will be included in a separate TAL
for k = 1:length(annotations)
    % Note: Each offset starts with '+' and finishes with unprintable ASCII '21'
    data = [data, sprintf('%s%s%s', '+', num2str(annotations(k).offset), 21)];
    charsToWrite = charsToWrite + length(num2str(annotations(k).offset)) + 2;
    % Note: Each duration finishes with unprintable ASCII '20'
    data = [data, sprintf('%s%s', num2str(annotations(k).duration), 20)];
    charsToWrite = charsToWrite + length(num2str(annotations(k).duration)) + 1;
    % Note: Each annotation finishes with unprintable ASCII '20' and must not contain
    %       any '20' within. Also because we assume one TAL per annotation
    %       then the unprintable ASCII '0' is added to close the TAL
    data = [data, sprintf('%s%s', num2str(annotations(k).label), [20,0])];
    charsToWrite = charsToWrite + length(annotations(k).label) + 2;
end

%Note: Unused bytes of the 'EDF Annotations' signal in the remainder of the data record are also filled with 0-bytes
%TODO: Is this really necessary??  (because of 16bit encoding)
if (mod(charsToWrite, 2) > 0)
    data = [data, sprintf('%s', 0)];
    charsToWrite = charsToWrite + 1;
end

fid = fopen(filename, 'w', 'ieee-le');

if (fid == -1)
    disp('Error creating output EDF+ file');
    return;
end

general_header_size = 256; %bytes
one_signal_header_size = 256; %bytes

% Write edf

% FIXED HEADER
header.version = 0;
header.local_patient_identification = patId;
header.local_recording_identification = recId;
header.startdate_recording = startdate;
header.starttime_recording = starttime;
header.num_signals = 1;
header.num_bytes_header = general_header_size + one_signal_header_size;
header.reserved = 'EDF+C';
header.duration_data_record = 0; % Note we assume an 'Annotations only' EDF+ file
header.num_data_records = 1;

fprintf(fid, trimAndFillWithBlanks(num2str(header.version), 8));   % version
fprintf(fid, '%-80s', header.local_patient_identification);
fprintf(fid, '%-80s', header.local_recording_identification);
fprintf(fid, '%-8s', header.startdate_recording);
fprintf(fid, '%-8s', header.starttime_recording);
actualHeaderBytes = general_header_size + one_signal_header_size*header.num_signals;
if (ne(actualHeaderBytes, header.num_bytes_header))
    disp('EDFwriteSignal: Warning, num_bytes_header does not match the actual number of header bytes. Fixed!');
end
fprintf(fid, trimAndFillWithBlanks(num2str(actualHeaderBytes), 8));
fprintf(fid, '%-44s', header.reserved);
fprintf(fid, trimAndFillWithBlanks(num2str(header.num_data_records), 8));
fprintf(fid, trimAndFillWithBlanks(num2str(header.duration_data_record), 8));
fprintf(fid, trimAndFillWithBlanks(num2str(header.num_signals), 4));

% SIGNAL DEPENDENT HEADER
header.signals_info(1).label = 'EDF Annotations';
header.signals_info(1).transducer_type = '';
header.signals_info(1).physical_dimension = '';
header.signals_info(1).physical_min = -32768;
header.signals_info(1).physical_max = 32767;
header.signals_info(1).digital_min = -32768;
header.signals_info(1).digital_max = 32767;
header.signals_info(1).prefiltering = '';
header.signals_info(1).num_samples_datarecord = charsToWrite/2; % TODO: Check appropriateness of this value (determines how many annotations can be saved per block)
header.signals_info(1).reserved = '';

fprintf(fid, '%-16s', header.signals_info(1).label);
fprintf(fid, '%-80s', header.signals_info(1).transducer_type);
fprintf(fid, '%-8s', header.signals_info(1).physical_dimension);
fprintf(fid, trimAndFillWithBlanks(num2str(header.signals_info(1).physical_min), 8));
fprintf(fid, trimAndFillWithBlanks(num2str(header.signals_info(1).physical_max), 8));
fprintf(fid, trimAndFillWithBlanks(num2str(header.signals_info(1).digital_min), 8));
fprintf(fid, trimAndFillWithBlanks(num2str(header.signals_info(1).digital_max), 8));
fprintf(fid, '%-80s', header.signals_info(1).prefiltering);
fprintf(fid, trimAndFillWithBlanks(num2str(header.signals_info(1).num_samples_datarecord), 8));
fprintf(fid, '%-32s', header.signals_info(1).reserved);

% DATA WRITING
header_length = general_header_size + header.num_signals * one_signal_header_size;
current_position = ftell(fid); % in bytes

if ne(header_length, current_position)
    disp('something wrong could be happening');
end

%TODO: Fill signal accordingly
% Actual writing
% data = Dmin + (Dmax - Dmin) * (signal - Pmin)/(Pmax - Pmin);
% data = typecast(int16(data), 'uint16'); % From double to 16bit unsigned integers
% 
% fwrite(fid, data, 'uint16');
fwrite(fid, data, 'char');
statusok = (fclose(fid) == 0);
