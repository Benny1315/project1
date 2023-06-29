% Main program for displaying the graphs and 
% filtering using a Savitzki Golai filter

clear; close all;

% load the data from Exel
H2S = readmatrix('fileName.xslx');
% load the Wavelength ~ 189-289[nm]
Wavelength = readmatrix('wavelength.xlsx');

% Inserting the matrix into the SetTheData function in order 
% to get the normalized graphs for the Excel file.
DXXXby0ppm = SetTheData(H2S);
if (length(DXXXby0ppm) == 3)
[H2S100ppm, H2S300ppm,...
   H2S500ppm] = DXXXby0ppm{[1,2,3]};
elseif (length(DXXXby0ppm) == 5)
    [H2S100ppm, H2S300ppm,...
   H2S500ppm, H2S50ppm,...
   H2S200ppm] = DXXXby0ppm{[1,2,3,4,5]};
end

% Plots the garph
plot(Wavelength,H2S100ppm(:,:));
xlim([189.7,294.26]); ylim([-1,1.5]);
title('H2S - 100[m] - Signal Gain = 40 - 100ppm');
xlabel('Wavelength [nm]'); grid on;
ylabel('Intensity');

plot(Wavelength,H2S300ppm(:,:));
xlim([189.7,294.26]); ylim([-1,1.5]);
title('H2S - 100[m] - Signal Gain = 40 - 300ppm');
xlabel('Wavelength [nm]'); grid on;
ylabel('Intensity');

plot(Wavelength,H2S500ppm(:,:));
xlim([189.7,294.26]); ylim([-1,1.5]);
title('H2S - X[m] - Signal Gain = X - 500ppm');
xlabel('Wavelength [nm]'); grid on; 
ylabel('Intensity');


% Savitzky-Goaly filter 
a = sgolayfilt(H2S100ppm(:,:),6,51,[],2);   % order 6' frame length 51
plot(Wavelength,a,'LineWidth',2); 
grid on; title('S-G filtered, order=6, framelength=51');
xlim([189.7,294.26]); ylim([-1,1.5]);
xlabel('Wavelength [nm]');

b = sgolayfilt(H2S300ppm(:,:),6,51,[],2);   % order 6' frame length 51
plot(Wavelength,b,'LineWidth',2); 
grid on; title('S-G filtered, order=6, framelength=51');
xlim([189.7,294.26]); ylim([-1,1.5]);
xlabel('Wavelength [nm]');

c = sgolayfilt(H2S500ppm(:,:),6,51,[],2);   % order 6' frame length 51
plot(Wavelength,c,'LineWidth',2); 
grid on; title('S-G filtered, order=6, framelength=51');
xlim([189.7,294.26]); ylim([-1,1.5]);
xlabel('Wavelength [nm]');
