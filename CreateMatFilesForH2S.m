clear; close all;

H2S = readmatrix('H2S\28_02_2023-D-169212-H2S-100m-Gain80-0.8OD.xlsx');
Wavelength = 1:256;

DXXXby0ppm = SetTheDatamlx(H2S);
if (length(DXXXby0ppm) == 3)
[H2S100ppm, H2S300ppm,...
   H2S500ppm] = DXXXby0ppm{[1,2,3]};
elseif (length(DXXXby0ppm) == 5)
    [H2S100ppm, H2S300ppm,...
   H2S500ppm, H2S50ppm,...
   H2S200ppm] = DXXXby0ppm{[1,2,3,4,5]};
end

DXXXby0ppm = SetTheDatamlx(H2S);
if (length(DXXXby0ppm) == 3)
[H2S50ppm, H2S200ppm,...
   H2S400ppm] = DXXXby0ppm{[1,2,3]};
elseif (length(DXXXby0ppm) == 5)
    [H2S100ppm, H2S300ppm,...
   H2S500ppm, H2S50ppm,...
   H2S200ppm] = DXXXby0ppm{[1,2,3,4,5]};
end

% Plots without filtering
plot(Wavelength,H2S100ppm);
xlim([1 256]); ylim([-1,1.5]);
title('H2S 100ppm');
xlabel('Wavelength [nm]'); grid on;

plot(Wavelength,H2S300ppm);
xlim([1 256]); ylim([-1,1.5]);
title('H2S 300ppm');
xlabel('Wavelength [nm]'); grid on;

plot(Wavelength,H2S500ppm);
xlim([1 256]); ylim([-1,1.5]);
title('H2S 500ppm');
xlabel('Wavelength [nm]'); grid on;


H2S100mGain80 = struct('H2S100ppm' , H2S100ppm , 'H2S300ppm' , H2S300ppm ...
    , 'H2S500ppm' , H2S500ppm);

% Filtering and smoothdata 100/300/500
SG100ppm = sgolayfilt(H2S100ppm(:,:),6,51,[],2);
SG300ppm = sgolayfilt(H2S300ppm(:,:),6,51,[],2);
SG500ppm = sgolayfilt(H2S500ppm(:,:),6,51,[],2);


% Plots with filtering
plot(Wavelength,SG100ppm);
xlim([1 256]); ylim([-1,1.5]);
title('H2S 100ppm SG-filtering');
xlabel('Wavelength [nm]'); grid on;

plot(Wavelength,SG300ppm);
xlim([1 256]); ylim([-1,1.5]);
title('H2S 300ppm SG-filtering');
xlabel('Wavelength [nm]'); grid on;

plot(Wavelength,SG500ppm);
xlim([1 256]); ylim([-1,1.5]);
title('H2S 500ppm SG-filtering');
xlabel('Wavelength [nm]'); grid on;



H2S100mGain80 = struct('SG100ppm' , SG100ppm , 'SG300ppm' , SG300ppm ...
    , 'SG500ppm' , SG500ppm);

H2S100ppmNF = ...
[ H2S15mGain10.H2S100ppm ; H2S15mGain40.H2S100ppm ; H2S15mGain80.H2S100ppm ;...
  H2S45mGain10.H2S100ppm ; H2S45mGain40.H2S100ppm ; H2S45mGain80.H2S100ppm ;...
  H2S100mGain10.H2S100ppm ; H2S100mGain40.H2S100ppm ; H2S100mGain80.H2S100ppm ];

H2S300ppmNF = ...
[ H2S15mGain10.H2S300ppm ; H2S15mGain40.H2S300ppm ; H2S15mGain80.H2S300ppm ;...
  H2S45mGain10.H2S300ppm ; H2S45mGain40.H2S300ppm ; H2S45mGain80.H2S300ppm ;...
  H2S100mGain10.H2S300ppm ; H2S100mGain40.H2S300ppm ; H2S100mGain80.H2S300ppm ];

H2S500ppmNF = ...
[ H2S15mGain10.H2S500ppm ; H2S15mGain40.H2S500ppm ; H2S15mGain80.H2S500ppm ;...
  H2S45mGain10.H2S500ppm ; H2S45mGain40.H2S500ppm ; H2S45mGain80.H2S500ppm ;...
  H2S100mGain10.H2S500ppm ; H2S100mGain40.H2S500ppm ; H2S100mGain80.H2S500ppm ];
