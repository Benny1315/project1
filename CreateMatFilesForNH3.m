clear; close all;

NH3 = readmatrix('fileName.xlsx');
Wavelength = 1:256;
DXXXby0ppm = SetTheDatamlx(NH3);
if (length(DXXXby0ppm) == 3)
[NH3100ppm, NH3300ppm,...
   NH3500ppm] = DXXXby0ppm{[1,2,3]};
elseif (length(DXXXby0ppm) == 5)
    [NH3100ppm, NH3300ppm,...
   NH3500ppm, NH350ppm,...
   NH3200ppm] = DXXXby0ppm{[1,2,3,4,5]};
end

% Plots without filtering
plot(Wavelength,NH3100ppm);
xlim([1 256]); ylim([-1,1.5]);
title('NH3 100ppm');
xlabel('Wavelength [nm]'); grid on;

plot(Wavelength,NH3300ppm);
xlim([1 256]); ylim([-1,1.5]);
title('NH3 300ppm');
xlabel('Wavelength [nm]'); grid on;

plot(Wavelength,NH3500ppm);
xlim([1 256]); ylim([-1,1.5]);
title('NH3 500ppm');
xlabel('Wavelength [nm]'); grid on;

if (length(DXXXby0ppm) == 5)

    figure
    plot(Wavelength,NH350ppm);
    xlim([1 256]); ylim([-1,1.5]);
    title('NH3 50ppm');
    xlabel('Wavelength [nm]'); grid on;

    figure
    plot(Wavelength,NH3200ppm);
    xlim([1 256]); ylim([-1,1.5]);
    title('NH3 200ppm');
    xlabel('Wavelength [nm]'); grid on;

end

if (length(DXXXby0ppm) == 3)
    NH315mGain80 = struct('NH3100ppm' , NH3100ppm , 'NH3300ppm' , NH3300ppm ...
    , 'NH3500ppm' , NH3500ppm);
elseif (length(DXXXby0ppm) == 5)
    NH315mGain80 = struct('NH3100ppm' , NH3100ppm , 'NH3300ppm' , NH3300ppm ...
    , 'NH3500ppm' , NH3500ppm , 'NH350ppm' , NH350ppm , 'NH3200ppm' , NH3200ppm);
end

% Filtering and smoothdata
SG100ppm = sgolayfilt(NH3100ppm(:,:),6,51,[],2);
SG300ppm = sgolayfilt(NH3300ppm(:,:),6,51,[],2);
SG500ppm = sgolayfilt(NH3500ppm(:,:),6,51,[],2);

if (length(DXXXby0ppm) == 5)

    SG50ppm = sgolayfilt(NH350ppm(:,:),6,51,[],2);
    SG200ppm = sgolayfilt(NH3200ppm(:,:),6,51,[],2);

end


% Plots with filtering
plot(Wavelength,SG100ppm);
xlim([1 256]); ylim([-1,1.5]);
title('NH3 100ppm SG-filtering');
xlabel('Wavelength [nm]'); grid on;

plot(Wavelength,SG300ppm);
xlim([1 256]); ylim([-1,1.5]);
title('NH3 300ppm SG-filtering');
xlabel('Wavelength [nm]'); grid on;

plot(Wavelength,SG500ppm);
xlim([1 256]); ylim([-1,1.5]);
title('NH3 500ppm SG-filtering');
xlabel('Wavelength [nm]'); grid on;


if (length(DXXXby0ppm) == 5)

    figure
    plot(Wavelength(:),SG50ppm(:,:));
    xlim([1 256]); ylim([-1,1.5]);
    title('NH3 50ppm');
    xlabel('Wavelength [nm]'); grid on;

    figure
    plot(Wavelength(:),SG200ppm(:,:));
    xlim([1 256]); ylim([-1,1.5]);
    title('NH3 200ppm');
    xlabel('Wavelength [nm]'); grid on;

end

if (length(DXXXby0ppm) == 3)
NH315mGain80 = struct('SG100ppm' , SG100ppm , 'SG300ppm' , SG300ppm ...
    , 'SG500ppm' , SG500ppm);
elseif (length(DXXXby0ppm) == 5)
    NH3100mGain10 = struct('SG100ppm' , SG100ppm , 'SG300ppm' , SG300ppm ...
    , 'SG500ppm' , SG500ppm , 'SG50ppm' , SG50ppm , 'SG200ppm' , SG200ppm);
end

NH3100ppmNF = ...
[ NH315mGain10.NH3100ppm ; NH315mGain40.NH3100ppm ; NH315mGain80.NH3100ppm ;...
  NH345mGain10.NH3100ppm ; NH345mGain40.NH3100ppm ; NH345mGain80.NH3100ppm ;...
  NH3100mGain10.NH3100ppm ; NH3100mGain40.NH3100ppm ; NH3100mGain80.NH3100ppm ];

NH3300ppmNF = ...
[ NH315mGain10.NH3300ppm ; NH315mGain40.NH3300ppm ; NH315mGain80.NH3300ppm ;...
  NH345mGain10.NH3300ppm ; NH345mGain40.NH3300ppm ; NH345mGain80.NH3300ppm ;...
  NH3100mGain10.NH3300ppm ; NH3100mGain40.NH3300ppm ; NH3100mGain80.NH3300ppm ];

NH3500ppmNF = ...
[ NH315mGain10.NH3500ppm ; NH315mGain40.NH3500ppm ; NH315mGain80.NH3500ppm ;...
  NH345mGain10.NH3500ppm ; NH345mGain40.NH3500ppm ; NH345mGain80.NH3500ppm ;...
  NH3100mGain10.NH3500ppm ; NH3100mGain40.NH3500ppm ; NH3100mGain80.NH3500ppm ];
