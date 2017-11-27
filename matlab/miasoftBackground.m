function [statusOK] = miasoftBackground(inputEDFname, confXMLName, outputEDFname)

% Temp, these should be added to the DLL generation tool
% addpath('../EDFlibrary');
% addpath('../xml');
% addpath('./3rdparty');
% addpath('./EEGarousals'); % Path to Isaac's EEG arousal detection algorithm
% addpath('./EOGanalysis'); % Path to EOG plots functionality

doSignalConditioning = 1; % Not tested with SHHS signals
doRespirationAnalysis = 0;
doUseMIASOFTeegArousals = 0; % Isaac's algorithm has not been tested with SHHS signals
doDebugHypnogram = 1;
doEOGplots = 1; % Generation of EOG rapidity/slowniness plots

statusOK = 0;

hdr = EDFreadHeader(inputEDFname);
if isempty(hdr)
    error('Error reading input EDF header');
end

% TODO: Is possible to simplify these steps?
[airflow, saturation, abdoRes, thorRes, eeg1, eeg2, emg, eogl, eogr, configuration, position, lightSignal, sleepStages, grade, registro] = inicioMIASOFT();

% Find appropriate montage in XML configuration file
confXML = xml_parseany(confXMLName);
indxMontage = -1;
for k = 1:length(confXML.montage)
   if matchMontage(confXML.montage(k), hdr)
       indxMontage = k;
       break;
   end
end
if (indxMontage == -1)
    % We skip this recording
    %error('Not able to found a valid montage');
    fprintf('\nNot able to found a valid montage\n');
    return;
end
configuration.device = confXML.montage(indxMontage).device;

% Read neurophysiological signals
[eeg1.Signal, configuration.refChannels.eeg1, eeg1.Label] = readSignal(inputEDFname, confXML, indxMontage, 'eeg1');
[eeg2.Signal, configuration.refChannels.eeg2, eeg2.Label] = readSignal(inputEDFname, confXML, indxMontage, 'eeg2');
[emg.Signal, configuration.refChannels.emg, emg.Label] = readSignal(inputEDFname, confXML, indxMontage, 'emg');
[eogl.Signal, configuration.refChannels.eogl, eogl.Label] = readSignal(inputEDFname, confXML, indxMontage, 'eog1');
[eogr.Signal, configuration.refChannels.eogr, eogr.Label] = readSignal(inputEDFname, confXML, indxMontage, 'eog2');

% Initialize structures
[eeg1, eeg2, emg, eogl, eogr, configuration] = initStructuresSleep(hdr, eeg1, eeg2, emg, eogl, eogr, configuration);

% Signal conditioning (not included in MIASOFT GUI version!!!)
if doSignalConditioning
    [ecg.Signal, configuration.refChannels.ecg, ecg.Label] = readSignal(inputEDFname, confXML, indxMontage, 'ecg');
    
    ecg.SampleRate = hdr.signals_info(configuration.refChannels.ecg).sample_rate;
    [eeg1, eeg2, emg, eogl, eogr] = sleepSignalPreprocessing(eeg1, eeg2, emg, eogl, eogr, ecg);
    clear ecg;
end

% Sleep analysis
[~, ~, emg, ~, ~, sleepStages, MembGrades, reasoningUnitsDetection] = analisisSignals(eeg1, eeg2, emg, eogl, eogr, sleepStages, grade, [], [], configuration.idioma);

if doDebugHypnogram
    % Debug analysis for hypnogram
    confidenceMatrix = zeros(5, length(sleepStages.Signal));
    for k = 1:length(sleepStages.Signal)
        epochStart = (k - 1) * configuration.EPOCH_DURATION + 1; % In seconds
        epochEnd = k * configuration.EPOCH_DURATION;

        confidenceMatrix(:, k) = [mean(MembGrades.F0(epochStart:epochEnd)), mean(MembGrades.F1(epochStart:epochEnd)), ...
            mean(MembGrades.F2(epochStart:epochEnd)), mean(MembGrades.FSP(epochStart:epochEnd)), mean(MembGrades.FR(epochStart:epochEnd))];
    end
    
    if isempty(outputEDFname)
        [pathstr,name] = fileparts(inputEDFname);
    else
        [pathstr,name] = fileparts(outputEDFname); 
    end
    outputEDFdebugName =  fullfile(pathstr, [name,'_HYPDEBUG.mat']);
        
    save(outputEDFdebugName, 'sleepStages', 'confidenceMatrix');
end

% 1. Build annotations structure
annotations = [];

% 1.1. Export hypnogram
annCount = 0;
sStageStr = '';
for k = 1:length(sleepStages.Signal) 
    if strcmpi(intToEDFplusSleepStage(sleepStages.Signal(k)), sStageStr)
        annotations(annCount).duration = annotations(annCount).duration + configuration.EPOCH_DURATION; % In seconds
    else
        annCount = annCount + 1;
        sStageStr = intToEDFplusSleepStage(sleepStages.Signal(k));
        annotations(annCount).offset = (k-1)*configuration.EPOCH_DURATION;
        annotations(annCount).duration = configuration.EPOCH_DURATION; % In seconds
        annotations(annCount).label = sStageStr;
    end
end

% 1.2. Export EOG analysis plots
if doEOGplots
    % PreCD: eog is already the desired EOG derivation (better sort E2-E1 horizontal), 
    %        which has been already Notch and ECG-artifact filtered
    
    % Configuration parameters
    intTime = 0.05;
        
    % Analysis of saturation intervals
    relEOG = eogReliabilityCheck(eogl.Signal, eogl.SampleRate);
    
    % Caculation of rapidity/slowniness features
    [rapidity, slowniness] = eogDetection(eogl.Signal, eogl.SampleRate, relEOG, intTime);
    
    % Signal downsampling (In principle rapidity and slowniness do only output
    % one sample each 30s epoch)
    srout = 1; % In Hz
    if (eogl.SampleRate < srout)
        error('Output requested sampling frequency cannot be satisfied');
    end
    rapidity = rapidity(1:eogl.SampleRate/srout:end);
    slowniness = slowniness(1:eogl.SampleRate/srout:end);
    
    % Write results back to preferred location
    if isempty(outputEDFname)
        [pathstr,name] = fileparts(inputEDFname);
    else
        [pathstr,name] = fileparts(outputEDFname); 
    end
    
    outputEDF = fullfile(pathstr, [name,'_eogPlotREM.EDF']);

    % Write header and signal back
    disp('Writing EOG rapidity header and signal back...');
    signal2EDF(rapidity, srout, hdr.local_patient_identification, hdr.local_recording_identification, ...
        'EOG Saccades', hdr.startdate_recording, hdr.starttime_recording, 0, ...
        max(rapidity), hdr.signals_info(configuration.refChannels.eogl).digital_min, hdr.signals_info(configuration.refChannels.eogl).digital_max, ...
        'u', hdr.signals_info(configuration.refChannels.eogl).prefiltering, outputEDF);
    disp('done!');

    outputEDF = fullfile(pathstr, [name,'_eogPlotSEM.EDF']);

    % Write header and signal back
    disp('Writing EOG slowniness header and signal back...');
    signal2EDF(slowniness, srout, hdr.local_patient_identification, hdr.local_recording_identification, ...
        'EOG Sem', hdr.startdate_recording, hdr.starttime_recording, 0, ...
        max(slowniness), hdr.signals_info(configuration.refChannels.eogl).digital_min, hdr.signals_info(configuration.refChannels.eogl).digital_max, ...
        'u', hdr.signals_info(configuration.refChannels.eogl).prefiltering, outputEDF);
    disp('done!');
    
    % Clean memory
    clear rapidity slowniness relEOG;
    
end
clear eogl eogr;

% 1.3. Export EEG arousals
if doUseMIASOFTeegArousals
    for k = 1:length(reasoningUnitsDetection.Arousals.OutPut)
        annCount = annCount + 1;
        annotations(annCount).offset = reasoningUnitsDetection.Arousals.OutPut(k).startSecond;
        annotations(annCount).duration = reasoningUnitsDetection.Arousals.OutPut(k).endSecond - reasoningUnitsDetection.Arousals.OutPut(k).startSecond;
        annotations(annCount).label = ['EEG arousal', '@@', eeg1.Label];
    end
else
    AnnotationSignalLabel = eeg1.Label;
     
    % TODO: Remove debug code when algorithm is stable
    debugArousals = annotations;
    
    % EEG arousals analysis (using method developed with Isaac)
    eegSignal.raw = eeg1.Signal;
    eegSignal.rate = eeg1.SampleRate;
    emgSignal.raw = emg.Signal;
    emgSignal.rate = emg.SampleRate;
    
    % Clean memory
    eeg1 = [];
    eeg2 = [];
        
    [arousals, removedArousals, spindles, ~, ~] = arousalDetection(eegSignal, emgSignal, sleepStages.Signal);
    
    eegSignal = [];
    emgSignal = [];
       
    for k = 1:length(arousals)
        annCount = annCount + 1;
        annotations(annCount).offset = arousals(k).start;
        annotations(annCount).duration = arousals(k).duration;
        annotations(annCount).label = ['EEG arousal', '@@', AnnotationSignalLabel];
        
        debugArousals(annCount) = annotations(annCount);
        if strcmp(arousals(k).source, 'alpha')
            debugArousals(annCount).label = ['Alpha arousal', '@@', AnnotationSignalLabel];
        end
    end
    
    debugCount = annCount;
    for k = 1:length(spindles)
        debugCount = debugCount + 1;
        debugArousals(debugCount).offset = spindles(k).start;
        debugArousals(debugCount).duration = spindles(k).duration;
        debugArousals(debugCount).label = ['Spindle activity', '@@', AnnotationSignalLabel];
    end
    
    if isempty(outputEDFname)
        [pathstr,name] = fileparts(inputEDFname);
    else
        [pathstr,name] = fileparts(outputEDFname); 
    end
    outputEDFdebugName =  fullfile(pathstr, [name,'_DEBUGAROUSAL.EDF']);
    
    statusOK = annotations2EDFplus(debugArousals, hdr.local_patient_identification, hdr.local_recording_identification, ...
        hdr.startdate_recording, hdr.starttime_recording, outputEDFdebugName);
    if not(statusOK)
        error('Problem exporting EEG arousal debug annotations');
    end
end

% Clean memory
eeg1 = [];
eeg2 = [];


% RESPIRATION ANALYSIS
if doRespirationAnalysis

    % Read respiratory signals
    [airflow.Signal, configuration.refChannels.airflow, airflow.Label] = readSignal(inputEDFname, confXML, indxMontage, 'airflow');
    [thorRes.Signal, configuration.refChannels.thorRes, thorRes.Label] = readSignal(inputEDFname, confXML, indxMontage, 'thorRes');
    [abdoRes.Signal, configuration.refChannels.abdoRes, abdoRes.Label] = readSignal(inputEDFname, confXML, indxMontage, 'abdoRes');
    [saturation.Signal, configuration.refChannels.saturation, saturation.Label] = readSignal(inputEDFname, confXML, indxMontage, 'saturation');
    [position.Signal, configuration.refChannels.position, position.Label] = readSignal(inputEDFname, confXML, indxMontage, 'position');
  
    % Resample Airflow, AbdoRes and ThorRes to 10 Hz (Original setting when designing MIASOFT)
    % Note: Most problematic issue here is the possibility that the three
    % signals have different sampling rates. Further review of the code should
    % be done to safely allow this possibility.
    RespSR = 10; % In Hz
    upFactor = lcm(RespSR, hdr.signals_info(configuration.refChannels.airflow).sample_rate)/max(hdr.signals_info(configuration.refChannels.airflow).sample_rate, RespSR);
    downFactor = lcm(RespSR, hdr.signals_info(configuration.refChannels.airflow).sample_rate)/min(hdr.signals_info(configuration.refChannels.airflow).sample_rate, RespSR);
    airflow.Signal = resample(airflow.Signal, upFactor, downFactor);
    hdr.signals_info(configuration.refChannels.airflow).sample_rate = RespSR;
    upFactor = lcm(RespSR, hdr.signals_info(configuration.refChannels.abdoRes).sample_rate)/max(hdr.signals_info(configuration.refChannels.abdoRes).sample_rate, RespSR);
    downFactor = lcm(RespSR, hdr.signals_info(configuration.refChannels.abdoRes).sample_rate)/min(hdr.signals_info(configuration.refChannels.abdoRes).sample_rate, RespSR);
    abdoRes.Signal = resample(abdoRes.Signal, upFactor, downFactor);
    hdr.signals_info(configuration.refChannels.abdoRes).sample_rate = RespSR;
    upFactor = lcm(RespSR, hdr.signals_info(configuration.refChannels.thorRes).sample_rate)/max(hdr.signals_info(configuration.refChannels.thorRes).sample_rate, RespSR);
    downFactor = lcm(RespSR, hdr.signals_info(configuration.refChannels.thorRes).sample_rate)/min(hdr.signals_info(configuration.refChannels.thorRes).sample_rate, RespSR);
    thorRes.Signal = resample(thorRes.Signal, upFactor, downFactor);
    hdr.signals_info(configuration.refChannels.thorRes).sample_rate = RespSR;

    [airflow, saturation, abdoRes, thorRes, position, lightSignal] = initStructuresRespiration(hdr, airflow, saturation, abdoRes, thorRes, position, lightSignal, configuration);

    [airflow, saturation, abdoRes, thorRes, artifacts] = artifactDetection(airflow, saturation, abdoRes, thorRes, configuration);

    [airflow, saturation, abdoRes, thorRes, reasoningUnitsDetection] = eventDetection(airflow, saturation, abdoRes, thorRes, emg, artifacts, configuration, position, lightSignal, sleepStages, reasoningUnitsDetection, []);
    [reasoningUnitsDetection, configuration] = fuzzyReasoning(reasoningUnitsDetection, configuration, []);
    [abdoRes, thorRes, reasoningUnitsClassification] = eventClassification(abdoRes, thorRes, reasoningUnitsDetection, artifacts, configuration, []);
    [reasoningUnitsClassification, configuration] = fuzzyReasoningClassification9inputs(reasoningUnitsClassification, configuration, []);
    lista = find([reasoningUnitsDetection.OutPut.indexConfirmed]~=-1);
    for j=1:length(lista)
        reasoningUnitsDetection.OutPut(lista(j)).indexClass = reasoningUnitsClassification(j);
    end

    % 1.3. Export respiratory events
    AnnotationSignalLabel = airflow.Label;
    for k = 1:length(lista)
        annCount = annCount + 1;
        event = reasoningUnitsDetection.OutPut(lista(k));
        annotations(annCount).offset = event.startSecond;
        annotations(annCount).duration = event.endSecond - event.startSecond;
        if (event.conf_apnea > event.conf_hypo)
            [y, pos] = max([event.indexClass.conf_central, event.indexClass.conf_obs, event.indexClass.conf_mixta]);
            switch pos
                case 1,
                    annotations(annCount).label = ['Central apnea', '@@', AnnotationSignalLabel];
                case 2,
                    annotations(annCount).label = ['Obstrucive apnea', '@@', AnnotationSignalLabel];
                case 3,
                    annotations(annCount).label = ['Mixed apnea', '@@', AnnotationSignalLabel];
                otherwise
                    annotations(annCount).label = ['Apnea', '@@', AnnotationSignalLabel];
            end
        else
            annotations(annCount).label = ['Hypopnea', '@@', AnnotationSignalLabel];
        end    
    end

    % 1.4. Export desaturations
    AnnotationSignalLabel = saturation.Label;
    % Note in MCH only desaturations >= 3% are considered
    DesatThreshold = 2.5;
    for k = 1:length(reasoningUnitsDetection.OutPut)
        if (reasoningUnitsDetection.OutPut(k).conf_eventSat > 0.5)
            if (reasoningUnitsDetection.OutPut(k).relatedDesat ~= -1)
                eventSat = reasoningUnitsDetection.Saturation(reasoningUnitsDetection.OutPut(k).relatedDesat);
                if (eventSat.desat_reduction >= DesatThreshold)
                    annCount = annCount + 1;
                    annotations(annCount).offset = eventSat.desat_startSecond;
                    annotations(annCount).duration = eventSat.desat_endSecond - eventSat.desat_startSecond;
                    annotations(annCount).label = ['Desaturation', '@@', AnnotationSignalLabel];
                end
            end
        end
    end
end
    
% 2. Export to EDF+ file
if isempty(outputEDFname)
    [pathstr,name,ext] = fileparts(inputEDFname);
    outputEDFname =  fullfile(pathstr, [name,'_MIASOFT.EDF']);
end
statusOK = annotations2EDFplus(annotations, hdr.local_patient_identification, hdr.local_recording_identification, ...
    hdr.startdate_recording, hdr.starttime_recording, outputEDFname);


% --- Rellena las estructuras de las señales con la informacion del edf.
function [eeg1, eeg2, emg, eogl, eogr, configuration] = initStructuresSleep(header, eeg1, eeg2, emg, eogl, eogr, configuration)

%EEG1
eeg1.NSamples = length(eeg1.Signal);
eeg1.SampleRate = header.signals_info(configuration.refChannels.eeg1).sample_rate;    
eeg1.physMin = header.signals_info(configuration.refChannels.eeg1).physical_min;       
eeg1.physMax = header.signals_info(configuration.refChannels.eeg1).physical_max;       
eeg1.duration = header.duration_data_record * header.num_data_records;    
eeg1.Samples_epoch = configuration.EPOCH_DURATION * eeg1.SampleRate;  
eeg1.Total_epochs = eeg1.NSamples / eeg1.Samples_epoch;
eeg1.Axes.min_y = eeg1.physMin;
eeg1.Axes.max_y = eeg1.physMax;
eeg1.Marca_medico = zeros(1, length(eeg1.Signal), 'uint8');

%EEG2
eeg2.NSamples = length(eeg2.Signal);
eeg2.SampleRate = header.signals_info(configuration.refChannels.eeg2).sample_rate; 
eeg2.physMin = header.signals_info(configuration.refChannels.eeg2).physical_min;   
eeg2.physMax = header.signals_info(configuration.refChannels.eeg2).physical_max;   
eeg2.duration = header.duration_data_record * header.num_data_records;
eeg2.Samples_epoch = configuration.EPOCH_DURATION * eeg2.SampleRate;  
eeg2.Total_epochs = eeg2.NSamples / eeg2.Samples_epoch;
eeg2.Axes.min_y = eeg2.physMin;
eeg2.Axes.max_y = eeg2.physMax;
eeg2.Marca_medico = zeros(1, length(eeg2.Signal), 'uint8');

%EMG
emg.NSamples = length(emg.Signal);
emg.SampleRate = header.signals_info(configuration.refChannels.emg).sample_rate;   
emg.physMin = header.signals_info(configuration.refChannels.emg).physical_min;     
emg.physMax = header.signals_info(configuration.refChannels.emg).physical_max;     
emg.duration = header.duration_data_record * header.num_data_records; 
emg.Samples_epoch = configuration.EPOCH_DURATION * emg.SampleRate;  
emg.Total_epochs = emg.NSamples / emg.Samples_epoch;
emg.Axes.min_y = emg.physMin;
emg.Axes.max_y = emg.physMax;
emg.Marca_medico = zeros(1, length(emg.Signal), 'uint8');

%EOGL
eogl.NSamples = length(eogl.Signal);
eogl.SampleRate = header.signals_info(configuration.refChannels.eogl).sample_rate;
eogl.physMin = header.signals_info(configuration.refChannels.eogl).physical_min;  
eogl.physMax = header.signals_info(configuration.refChannels.eogl).physical_max;  
eogl.duration = header.duration_data_record * header.num_data_records;
eogl.Samples_epoch = configuration.EPOCH_DURATION * eogl.SampleRate;  
eogl.Total_epochs = eogl.NSamples / eogl.Samples_epoch;
eogl.Axes.min_y = eogl.physMin;
eogl.Axes.max_y = eogl.physMax;
eogl.Marca_medico = zeros(1, length(eogl.Signal), 'uint8');

%EOGR
eogr.NSamples = length(eogr.Signal);
eogr.SampleRate = header.signals_info(configuration.refChannels.eogr).sample_rate; 
eogr.physMin = header.signals_info(configuration.refChannels.eogr).physical_min;   
eogr.physMax = header.signals_info(configuration.refChannels.eogr).physical_max;   
eogr.duration = header.duration_data_record * header.num_data_records;
eogr.Samples_epoch = configuration.EPOCH_DURATION * eogr.SampleRate;  
eogr.Total_epochs = eogr.NSamples / eogr.Samples_epoch;
eogr.Axes.min_y = eogr.physMin;
eogr.Axes.max_y = eogr.physMax;
eogr.Marca_medico = zeros(1, length(eogr.Signal), 'uint8');

% Configuration
configuration.recordingDuration = header.duration_data_record * header.num_data_records;
configuration.Axes.limite = configuration.recordingDuration;
configuration.sesion = 1;

% --- Rellena las estructuras de las señales con la informacion del edf.
function [airflow, saturation, abdoRes, thorRes, position, lightSignal] = initStructuresRespiration(header, airflow, saturation, abdoRes, thorRes, position, lightSignal, configuration)

% Airflow
airflow.NSamples = length(airflow.Signal);
airflow.SampleRate = header.signals_info(configuration.refChannels.airflow).sample_rate;     % Frecuencia
airflow.physMin = header.signals_info(configuration.refChannels.airflow).physical_min;       % Valor minimo
airflow.physMax = header.signals_info(configuration.refChannels.airflow).physical_max;       % Valor maximo
airflow.duration = header.duration_data_record * header.num_data_records;       % Duracion (Segundos )
airflow.Samples_epoch = configuration.EPOCH_DURATION * airflow.SampleRate;      % Numero de datos por epoch
airflow.Total_epochs = airflow.NSamples / airflow.Samples_epoch;
airflow.Marca_medico = zeros(1, length(airflow.Signal), 'uint8');
airflow.Axes.min_y = airflow.physMin;
airflow.Axes.max_y = airflow.physMax;

% Saturation. Comprobamos que la frecuencia sea 1 Hz. Para
% la saturacion no tiene sentido (de momento) tener mas datos
% por segundo.
srSat = header.signals_info(configuration.refChannels.saturation).sample_rate;
if srSat>1
    saturation.Signal = saturation.Signal(1:srSat:end);
    saturation.NSamples = length(saturation.Signal);
    saturation.SampleRate = 1;
else
    saturation.NSamples = length(saturation.Signal);
    saturation.SampleRate = srSat;
end
saturation.physMin = header.signals_info(configuration.refChannels.saturation).physical_min; 
saturation.physMax = header.signals_info(configuration.refChannels.saturation).physical_max;
saturation.duration = header.duration_data_record * header.num_data_records;     
saturation.Samples_epoch = configuration.EPOCH_DURATION * saturation.SampleRate; 
saturation.Total_epochs = saturation.NSamples / saturation.Samples_epoch;
saturation.Axes.min_y = saturation.physMin;
saturation.Axes.max_y = saturation.physMax;
saturation.Marca_medico = zeros(1, length(saturation.Signal), 'uint8');

%AbdoRes
abdoRes.NSamples = length(abdoRes.Signal);
abdoRes.SampleRate = header.signals_info(configuration.refChannels.abdoRes).sample_rate;   
abdoRes.physMin = header.signals_info(configuration.refChannels.abdoRes).physical_min;     
abdoRes.physMax = header.signals_info(configuration.refChannels.abdoRes).physical_max;     
abdoRes.duration = header.duration_data_record * header.num_data_records;     
abdoRes.Samples_epoch = configuration.EPOCH_DURATION * abdoRes.SampleRate;  
abdoRes.Total_epochs = abdoRes.NSamples / abdoRes.Samples_epoch;
abdoRes.Axes.min_y = abdoRes.physMin;
abdoRes.Axes.max_y = abdoRes.physMax;
abdoRes.Marca_medico = zeros(1, length(abdoRes.Signal), 'uint8');

%ThorRes
thorRes.NSamples = length(thorRes.Signal);
thorRes.SampleRate = header.signals_info(configuration.refChannels.thorRes).sample_rate;
thorRes.physMin = header.signals_info(configuration.refChannels.thorRes).physical_min;   
thorRes.physMax = header.signals_info(configuration.refChannels.thorRes).physical_max;   
thorRes.duration = header.duration_data_record * header.num_data_records;   
thorRes.Samples_epoch = configuration.EPOCH_DURATION * thorRes.SampleRate;  
thorRes.Total_epochs = thorRes.NSamples / thorRes.Samples_epoch;
thorRes.Axes.min_y = thorRes.physMin;
thorRes.Axes.max_y = thorRes.physMax;
thorRes.Marca_medico = zeros(1, length(thorRes.Signal), 'uint8');

%Position
srPos = header.signals_info(configuration.refChannels.position).sample_rate;
if srPos>1
    position.Signal = position.Signal(1:srPos:end);
    position.NSamples = length(position.Signal);
    position.SampleRate = 1;
else
    position.NSamples = length(position.Signal);
    position.SampleRate = srPos;
end
if not(strcmpi(configuration.device, 'SHHS'))
    position.Signal = importSomnomedicsPosition(position.Signal, configuration.device);
end
% Para que se vean bien en el axe
position.Axes.min_y = -1;
position.Axes.max_y = 4;
position.physMin = header.signals_info(configuration.refChannels.position).physical_min;       
position.physMax = header.signals_info(configuration.refChannels.position).physical_max;       
position.duration = header.duration_data_record * header.num_data_records;        
position.Samples_epoch = configuration.EPOCH_DURATION * position.SampleRate;  
position.Total_epochs = position.NSamples / position.Samples_epoch;

% Light
if strcmpi(configuration.device, 'SHHS')
    %lightSignal.Signal = signals.luz;
    lightSignal.NSamples = length(lightSignal.Signal);
    lightSignal.SampleRate = header.signals_info(configuration.refChannels.lightSignal).sample_rate;     
    lightSignal.physMin = header.signals_info(configuration.refChannels.lightSignal).physical_min;       
    lightSignal.physMax = header.signals_info(configuration.refChannels.lightSignal).physical_max;       
    lightSignal.duration = header.duration_data_record * header.num_data_records;   
    lightSignal.Samples_epoch = configuration.EPOCH_DURATION * lightSignal.SampleRate;  
    lightSignal.Total_epochs = lightSignal.NSamples / lightSignal.Samples_epoch;
else
    % E.g. Somno1020
    lightSignal.Signal = ones(header.duration_data_record * header.num_data_records, 1);
    lightSignal.NSamples = length(lightSignal.Signal);
    lightSignal.SampleRate = 1;
    lightSignal.physMin = 0;       
    lightSignal.physMax = 1;    
    lightSignal.duration = length(lightSignal.Signal);  
    lightSignal.Samples_epoch = configuration.EPOCH_DURATION * lightSignal.SampleRate;  
    lightSignal.Total_epochs = lightSignal.NSamples / lightSignal.Samples_epoch;
end
% Para que se vean bien en el axe
lightSignal.Axes.min_y = lightSignal.physMin - 1;
lightSignal.Axes.max_y = lightSignal.physMax + 1;

function annotation = intToEDFplusSleepStage(intSleepStage)

switch(intSleepStage)
    case 0,
        annotation = 'Sleep stage W';
    case 1,
        annotation = 'Sleep stage N1';
    case 2,
        annotation = 'Sleep stage N2';
    case 3,
        annotation = 'Sleep stage N3';
    case 5,
        annotation = 'Sleep stage R';
    otherwise,
        annotation = 'Error';
end

function positionSignal = importSomnomedicsPosition(inputPositionSignal, deviceName)

switch deviceName
    case {'Somno1020', 'SomnoPlus'}
        positionSignal = ones(size(inputPositionSignal)) .* 4; % Unknown default
        positionSignal(inputPositionSignal >= -100 & inputPositionSignal <= 100) = 3;
        positionSignal(inputPositionSignal >= 500 & inputPositionSignal <= 700) = 9; % Movement (Upright)
        %positionSignal(inputPositionSignal >= 614 & inputPositionSignal <= 819) = 9; % Movement (Upside down)
        positionSignal(inputPositionSignal >= 1200 & inputPositionSignal <= 1400) = 1;
        %positionSignal(inputPositionSignal >= 1536 & inputPositionSignal <= 1740) = 1;
        positionSignal(inputPositionSignal >= 1900 & inputPositionSignal <= 2100) = 0;
        %positionSignal(inputPositionSignal >= 2252 & inputPositionSignal <= 2416) = 0;
        positionSignal(inputPositionSignal >= 2400 & inputPositionSignal <= 2600) = 9; % Movement (Upside down)
        %positionSignal(inputPositionSignal >= 2662 & inputPositionSignal <= 2867) = 2;
        positionSignal(inputPositionSignal >= 2900 & inputPositionSignal <= 3100) = 2;
    case {'SomnoPSG'}
        positionSignal = ones(size(inputPositionSignal)) .* 4; % Unknown default
        positionSignal(inputPositionSignal >= 0 & inputPositionSignal <= 100) = 3;
        positionSignal(inputPositionSignal >= 614 & inputPositionSignal <= 819) = 9; % Movement (Upside down)
        positionSignal(inputPositionSignal >= 1126 & inputPositionSignal <= 1331) = 1;
        %positionSignal(inputPositionSignal >= 1200 & inputPositionSignal <= 1400) = 1;
        positionSignal(inputPositionSignal >= 1536 & inputPositionSignal <= 1740) = 1;
        %positionSignal(inputPositionSignal >= 1900 & inputPositionSignal <= 2100) = 0;
        positionSignal(inputPositionSignal >= 1945 & inputPositionSignal <= 2150) = 0;
        positionSignal(inputPositionSignal >= 2252 & inputPositionSignal <= 2416) = 0;
        %positionSignal(inputPositionSignal >= 2400 & inputPositionSignal <= 2600) = 9; % Movement (Upside down)
        positionSignal(inputPositionSignal >= 2498 & inputPositionSignal <= 2621) = 9; % Movement (Upright)
        positionSignal(inputPositionSignal >= 2662 & inputPositionSignal <= 2867) = 2;
        %positionSignal(inputPositionSignal >= 2900 & inputPositionSignal <= 3100) = 2;
    case 'CHUAC'
        positionSignal = inputPositionSignal;
        positionSignal(inputPositionSignal<=200) = 2; % Supine
        positionSignal(inputPositionSignal>200 & inputPositionSignal<=600) = 1; % Left
        positionSignal(inputPositionSignal>600 & inputPositionSignal<=800) = 0; % Right
        positionSignal(inputPositionSignal>800) = 3; % Prone
    otherwise
        positionSignal = inputPositionSignal;
end

function [result] = matchMontage(xmlMontageId, hdr)

result = 1;

channelMapping = xmlMontageId.channelmapping;

signalsToCheck = {'eeg1', 'eeg2', 'emg', 'eog1', 'eog2', 'airflow', 'abdoRes', 'thorRes', 'saturation', 'position'};

for k = 1:length(signalsToCheck)
    signal = signalsToCheck{k};
    for k1 = 1:length(channelMapping.(signal).channel)
        channel = xmlMontageId.channelmapping.(signal).channel(k1);
        label = channel.ATTRIBUTE.label;
        idxChannel = channel.ATTRIBUTE.idx;
        result = result && (findIndexChannelByLabel(hdr, label) == idxChannel);
    end
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

label = channelMapping.(signalName).ATTRIBUTE.label;
        
