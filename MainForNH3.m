% Main program for displaying the graphs and 
% filtering using a Savitzki Golai filter

clear; close all;

% load the data from Exel
NH3 = readmatrix('fileName.xslx');
% load the Wavelength ~ 189-289[nm]
Wavelength = readmatrix('wavelength.xlsx');


% Inserting the matrix into the SetTheData function in order 
% to get the normalized graphs for the Excel file.
% Provided that the order of the data in Excel is in this order.
% In terms of gas concentration.
DXXXby0ppm = SetTheData(NH3);
if (length(DXXXby0ppm) == 3)
[NH3100ppm, NH3300ppm,...
   NH3500ppm] = DXXXby0ppm{[1,2,3]};
elseif (length(DXXXby0ppm) == 5)
    [NH3100ppm, NH3300ppm,...
   NH3500ppm, NH350ppm,...
   NH3200ppm] = DXXXby0ppm{[1,2,3,4,5]};
end

% Plots the garph
plot(Wavelength,NH3100ppm(:,:));
xlim([189.7,294.26]); ylim([-1,1.5]);
title('NH3 - X[m] - Signal Gain = X - 100ppm');
xlabel('Wavelength [nm]'); grid on;

plot(Wavelength,NH3300ppm(:,:));
xlim([189.7,294.26]); ylim([-1,1.5]);
title('NH3 - X[m] - Signal Gain = X - 300ppm');
xlabel('Wavelength [nm]'); grid on;


plot(Wavelength,NH3500ppm(:,:));
xlim([189.7,294.26]); ylim([-1,1.5]);
title('NH3 - X[m] - Signal Gain = X - 500ppm');
xlabel('Wavelength [nm]'); grid on;

if (length(DXXXby0ppm) == 5)

    figure
    plot(Wavelength,NH350ppm);
    xlim([189.7,294.26]); ylim([-1,1.5]);
    title('NH3  ---  50ppm');
    xlabel('Wavelength [nm]'); grid on;

    figure
    plot(Wavelength,NH3200ppm);
    xlim([189.7,294.26]); ylim([-1,1.5]);
    title('NH3  ---  200ppm');
    xlabel('Wavelength [nm]'); grid on;

end


% Savitzky-Goaly filter 
a = sgolayfilt(NH3100ppm(:,:),6,51,[],2);   % order 6' frame length 51
plot(Wavelength,a,'LineWidth',2); 
grid on; title('S-G filtered, order=6, framelength=51');
xlim([189.7,294.26]); ylim([-1,1.5]);
xlabel('Wavelength [nm]');

b = sgolayfilt(NH3300ppm(:,:),6,51,[],2);   % order 6' frame length 51
plot(Wavelength,b,'LineWidth',2); 
grid on; title('S-G filtered, order=6, framelength=51');
xlim([189.7,294.26]); ylim([-1,1.5]);
xlabel('Wavelength [nm]');

c = sgolayfilt(NH3500ppm(:,:),6,51,[],2);   % order 6' frame length 51
plot(Wavelength,c,'LineWidth',2); 
grid on; title('S-G filtered, order=6, framelength=51');
xlim([189.7,294.26]); ylim([-1,1.5]);
xlabel('Wavelength [nm]');

% If there are measurements of 200 and 50 in the file, you need to filter as well.
% Just add lines for them.


