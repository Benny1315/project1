%% Basad
close all; clear; 

desired_size = [ 105 , 105 ];  % Desired size of the PNG images

for i = 1:numel(filename(:,5))   % choose how many PNG file you want
    % Plot the data from the row
%    data_row = mean(H2S500ppm,1);
%    plot(data_row(7:248),'Color','k',LineWidth=2.2);
    data_row = H2S400ppm(i,7:248);  % remove the noisy signal pixel
    plot(data_row,'Color','k',LineWidth=2.2);

    % Set the same ylim and xlim
    ylim([-0.2 1.5]);
    xlim([1 numel(data_row)]);

    % Hide the x-label and y-label
    set(gca, 'Visible', 'off');

    % Set the size of the figure
    fig = gcf;  % Get the current figure handle
    fig.Units = 'pixels';
    fig.Position = [0 0 desired_size(2) desired_size(1)];

    % Set a white background inside the plot area
    ax = gca;  % Get the current axes handle
    ax.Color = 'white';

    % Capture the image of the figure
    frame = getframe(fig);
    img = frame.cdata;

    % Resize the image to the desired size
    resized_img = imresize(img, desired_size);

    % Generate a unique filename for each figure
    filename = sprintf('H2S_500ppmTEST_%d.png', i);

    % Save the image as a PNG file
    imwrite(resized_img, filename);

    % Close the figure to avoid unnecessary accumulation of open figures
    close(fig);
end
