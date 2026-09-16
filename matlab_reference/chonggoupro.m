function reconstructed = chonggoupro(phase0Path,phase120Path,phase240Path,outputPath)
%CHONGGOUPRO Reconstruct a FINCH magnitude image from three phase steps.
%   RECONSTRUCTED = CHONGGOUPRO(PHASE0PATH,PHASE120PATH,PHASE240PATH,
%   OUTPUTPATH) reproduces the MATLAB calculation used to validate the
%   Python implementation. All spatial units are millimetres.

theta1 = 0;
theta2 = 2*pi/3;
theta3 = 4*pi/3;
wavelength = 625e-6;
pixelSize = 3.45e-3;
propagationDistance = 4.2;

phase0 = double(imread(phase0Path));
phase120 = double(imread(phase120Path));
phase240 = double(imread(phase240Path));
assert(isequal(size(phase0),size(phase120),size(phase240)), ...
    'All three phase-shifted holograms must have the same dimensions.');

% Use the first channel when an image is stored as RGB grayscale data.
phase0 = phase0(:,:,1);
phase120 = phase120(:,:,1);
phase240 = phase240(:,:,1);

% Three-step phase-shifting demodulation.
complexHologram = ...
    phase0.*(exp(-1i*theta3)-exp(-1i*theta2)) ...
    + phase120.*(exp(-1i*theta1)-exp(-1i*theta3)) ...
    + phase240.*(exp(-1i*theta2)-exp(-1i*theta1));

[height,width] = size(complexHologram);
x = wavelength*((0:width-1)-width/2)/(pixelSize*width);
y = wavelength*((0:height-1)-height/2)/(pixelSize*height);
[gridX,gridY] = meshgrid(x,y);

waveNumber = 2*pi/wavelength;
transfer = exp(1i*waveNumber*propagationDistance ...
    .*sqrt(1-gridX.^2-gridY.^2));
centeredSpectrum = fftshift(fft2(complexHologram));
propagatedField = ifft2(centeredSpectrum.*transfer);
reconstructed = mat2gray(abs(propagatedField));

if nargin >= 4 && ~isempty(outputPath)
    imwrite(reconstructed,outputPath);
end
end
