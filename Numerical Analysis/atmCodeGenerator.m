%Generate shading coeffcients for Atmospheric Shader
%Code by Jose Miguel Santana

function generateCode()
clc

%Values of R, A and OD can be used to find a 
%proper fit to use in our shader
%Remember to rescale the Map to [0..1]
[R, A, OD] = createODMap();
gScale = max(OD(:)); %Maximum value of OD for Scaling

%Sampling Points %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nRaySamples = 21;
samplesFactor = 0.01;
samples = 1.0 - (exp(-linspace(0, samplesFactor, nRaySamples))-exp(-samplesFactor))*(1/(1-exp(-samplesFactor)));
samples2 = (exp(-linspace(0, samplesFactor, ceil(nRaySamples/2)))-exp(-samplesFactor))*(1/(1-exp(-samplesFactor)));

%Integration Coefficients %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Code for View Ray Sampling\n\n');

str = '';
str = addLine(str, sprintf('if (n == %d) return x0 + d * %f;\n', [0:nRaySamples-1; samples]));
str = addLine(str, '\n\n//-----\n\n');
str = addLine(str, sprintf('if (n == %d) return x0 * %f;\n', [0:ceil(nRaySamples/2)-2; samples2(1:length(samples2)-1)]));
str = addLine(str, sprintf('if (n == %d) return 0.0;', ceil(nRaySamples/2)-1));
str = addLine(str, sprintf('if (n == %d) return x1 * %f;\n', [0:ceil(nRaySamples/2)-2; samples2(length(samples2)-1:-1:1)]));

%Sampling Colors
fprintf('Code for Integration Process\n\n');

nW = 10;

Ns = 2.545e25;
n = 1.000278;

K = (2*pi^2*(n^2-1)^2)/(3*Ns);
w = 380e-9:(780-380) / (nW-1) * 1e-9:780e-9;

str = addLine(str, sprintf('const int nW = %d;', nW-1));
str = addLine(str, sprintf('highp vec3 CIElevels[%d];', nW-1));

XYZs = zeros(3,nW-1); %Wavelength integration
for i = 1:nW-1
    XYZs(:,i) = integrateSunIntensityToCIE1931(w(i), w(i+1));
    meanW = (w(i) + w(i+1)) / 2.0;
    kw4 = K * meanW^-4;
    XYZs(:,i) = XYZs(:,i) * kw4; %Adjusting to K/w^4
end
maxCIE = max(XYZs(:));
XYZs = XYZs / maxCIE; %Normalizing

for i = 1:nW-1
    str = addLine(str, sprintf('CIElevels[%d] = vec3(%e, %e, %e); //w = %.2f - %.2f', i-1, XYZs(1, i), XYZs(2, i), XYZs(3, i), w(i)*10^9, w(i+1)*10^9));
end

str = addLine(str, '');
str = addLine(str, sprintf('highp float I0CIEKw4Scale = %e;', maxCIE));
str = addLine(str, sprintf('highp float gScale4PiKw4[%d];', nW-1));

w4 = ((w(1:length(w)-1) + w(2:length(w))) / 2.0).^-4;
gScale4PiKw4 = 4 * pi * gScale * K * w4;
str = addLine(str, sprintf('gScale4PiKw4[%d] = %f;\n', [0:nW-2; gScale4PiKw4]));

fprintf(str);
clipboard('copy',str)

end

function s = addLine(s1, s2)
    s = sprintf('%s\n%s', s1, s2);
end

function XYZ = integrateSunIntensityToCIE1931(w1, w2) %W. in meters

n = 100;
ws = linspace(w1, w2, n);
XYZs = zeros(3,n);

h = 6.626070040e-34; %Plank const. Jxs
c = 299792458; %Speed of light mxs
k = 1.38064852e-23; % Boltzmann const. J/K
T = 5800; %Sun temperature K.
I = ((2*h*c^2) ./(ws.^5))./(exp((h*c)./(ws*k*T))-1);

for i = 1:n
    XYZs(:,i) = waveLengthToCIE1931(ws(i) * 10^9); %W in nm.
    
    XYZs(:,i) = XYZs(:,i) * I(i); %Sun intensity on that wavelength
end

XYZ = [0,0,0];
XYZ(1) = trapz(ws, XYZs(1,:)); %Working on nm.
XYZ(2) = trapz(ws, XYZs(2,:)); 
XYZ(3) = trapz(ws, XYZs(3,:));

end

function XYZ = waveLengthToCIE1931(w)
x = 1.065*exp(-0.5*((w-595.8)/33.33)^2) + 0.366*exp(-0.5*((w-446.8)/19.44)^2);
y = 1.014*exp(-0.5*((log(w)-log(556.3))/0.075)^2);
z = 1.839*exp(-0.5*((log(w)-log(449.8))/0.051)^2);
XYZ = [x y z];
end

%Optical Depth Map Generation
function [rmap, amap, odmap] = createODMap()
global sh;
sh = 6e9 + 5e4;
global er;
er = 6e9;
global atmThickness;
atmThickness = 5e4;

[rmap,amap]=meshgrid(linspace(er,sh,100),linspace(pi/2,pi,100)); %Row per angle value

odmap = rmap;
for i=1:size(rmap,1)
    for j=1:size(rmap,2)
        x = rmap(i,j)*cos(amap(i,j));
        y = rmap(i,j)*sin(amap(i,j));
        odmap(i,j) = opticalDepthOfRayFromInfinity(x,y);
    end
end

%Simplifying 
surf(rmap, amap, odmap);
end

function od = opticalDepthOfRayFromInfinity(x,y)
global sh;
n = 1000;

x0 = -sqrt(sh^2 - y^2);
xs = linspace(x0, x ,n);
ys = repmat(y, 1, n);

optDen = opticalDensityOfPoint(xs, ys);

od = trapz(xs,optDen) ; %Working on nm.
end

function optDen = opticalDensityOfPoint(xs, ys)
global er;
global atmThickness;
h = sqrt(xs.^2 + ys.^2) - er;%Heigths
h0 = (7994 * 5e4) / atmThickness;
optDen = exp(-( h / h0)); %Density coefficient

optDen(optDen < 0) = 0;
end
