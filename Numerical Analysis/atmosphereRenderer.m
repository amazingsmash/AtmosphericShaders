%Numerical analysis of full atmospheric Rayleight scattering
%Code by Jose Miguel Santana

function [worldImage, atmImage, atmIntensityImage] = atmosphereRenderer(imSize, nSamplingPoints, nws)

close all;

global nPointsPerRay;
global er;
global sh;
global atmThickness;
nPointsPerRay = nSamplingPoints;

%Scene parameters
sunDir = [1, 0, 0];
sunPos = 149.6e9 * (-sunDir / norm(sunDir));
er = 6e9; %Earth radius
atmThickness = 5e4;
sh = er + atmThickness; %Stratosphere radius

%Camera position
pv = [0, 0, -2]*er;
vd = -pv / norm(pv);
vd = vd / norm(vd);

up = [0, vd(3), vd(2)];

imageRays = pixelRaysForXYZNear(imSize, vd, up, -1,1,-1,1);

worldImage = earthImage(imageRays,pv);

[atmImage, atmIntensityImage] = atmosphereImage(imageRays, worldImage, sunPos, pv, nws);

if 1 %Show final image exaggerating color
    figure;
    atmFakeColor = atmImage / max(max(max(atmImage)));
    imshow(worldImage + atmFakeColor);
end

end

function imageRays = pixelRaysForXYZNear(imSize, vd, up, xmin, xmax, ymin, ymax)

[x,y] = meshgrid(linspace(xmin,xmax,imSize), linspace(ymin,ymax,imSize));
right = cross(up,vd);

imageRays = zeros(imSize, imSize, 3);
for i = 1:size(x,1)
    for j = 1:size(x,2)
        rayDir = vd + up*y(i,j) + right*x(i,j);
        imageRays(i,j,:) = rayDir;
    end
end

end

function image = earthImage(imageRays,pv)
global er;
worldTexture = im2double(imread('world.jpg'));

%Buffers
image= zeros(size(imageRays,1),size(imageRays,1),3);
origin = pv;

sunRadius = 696300*10^3;
for i = 1:size(imageRays,1)
    for j = 1:size(imageRays,2)
        rayDir = imageRays(i,j,:);
        rayDir = rayDir(:)';
        
        %Displaying Sun as a bright dot
        if distancePointToLine([0 0 0], origin, rayDir) < sunRadius
            image(i,j,:) = [1 1 1];
        end
        
        [paE,~]=raySphereIntersection(origin, rayDir, er);
        if ~isnan(paE)
            p = origin + rayDir * paE;
            image(i,j,:) = earthColorForPoint(worldTexture, p);
        end
    end
end


end

function c = earthColorForPoint(worldImage, p)
sunPos = [-1 0 0];
ilu = dot(p/norm(p), sunPos/norm(sunPos) );
if ilu < 0
    ilu = 0.2; %Ambient light
else
    ilu = 0.2 + 2*ilu;
end

worldImWidth = size(worldImage,2);
worldImHeight = size(worldImage,1);
[a,e] = cart2sph(p(1), p(3), p(2));

an = wrapTo2Pi(a-pi/2)/(2*pi);
en = wrapTo2Pi(e+(pi/2))/pi;

xwi = 1+round(an*worldImWidth);
ywi = 1+round(en*worldImHeight);
c = worldImage(ywi,xwi,:) * ilu;
end

function [atmColor, atmIntensityImage] = atmosphereImage(imageRays, earthImage, sunPos, pv, nws)

ws = linspace(380, 780, nws); %Visible spectre Wavelength in nanometers
wDistanceInMeters = (ws(2) - ws(1)) * 10^-9;
wis = sunWavelengthIntensity(ws, norm(sunPos)); %Received intensity

atmColor = zeros(size(earthImage,1), size(earthImage,2),3);
wColor = atmColor;

%Computing ray extreme points for every ray
origin = pv;

atmIntensityImage  = atmReflectedLightOnRayForWavelengths(origin,imageRays, ws, wis);

if 0 %Show interpolated colors
    showUsedColors(ws);
end

wDelta = ws(2)- ws(1);
for i = 1:length(ws)
    
    rgb = waveLengthToRGB(ws(i), wDelta);
    
    wColor(:,:,1) = atmIntensityImage(:,:,i) * rgb(1);
    wColor(:,:,2) = atmIntensityImage(:,:,i) * rgb(2);
    wColor(:,:,3) = atmIntensityImage(:,:,i) * rgb(3);
    
    atmColor = atmColor + wColor;
end

atmColor = atmColor * wDistanceInMeters;

end

function showUsedColors(ws)
colorsImage = zeros(length(ws)*40,40,3);
for i = 1:length(ws)
    rgb = waveLengthToRGB(ws(i));
    
    for j = 1:40
        for k = 1:40
            colorsImage(i*40+j,k,:) = rgb;
        end
    end
end

figure;
imshow(colorsImage);
title(sprintf('%.3f, ', ws));
end

function I = sunWavelengthIntensity(v, distance)
h = 6.626070040e-34; %Plank const. Jxs
c = 299792458; %Speed of light mxs
k = 1.38064852e-23; % Boltzmann const. J/K

T = 5800; %Sun temperature K.

I = ((2*h*c^2) ./(v.^5))./(exp((h*c)./(v*k*T))-1);

I = I / (distance^2); %Received intensity
end

function [pa,pb] = raySphereIntersection(o, d, r)
% http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
a = dot(d,d);
b = 2.0 * dot(o,d);
c = dot(o,o) - (r*r);
q = (b*b) - 4.0 * a * c;

if q > 0.0
    sq = sqrt(q);
    t1 = (-b - sq) / (2.0*a);
    t2 = (-b + sq) / (2.0*a);
    
    if (t1 < 0.0 && t2 < 0.0)
        pa = NaN;
        pb = NaN;
        return;
    end
    
    pa = max(min(t1,t2), 0.0);
    pb = max(t1,t2);
else
    pa = NaN;
    pb = NaN;
    return;
end

end

function [x0, x1, y0, m] = simplifyRay(a, b)
ab = (b-a)/norm(b-a);
c = a + dot(-a,ab) * ab;

if norm(c) == 0 || sum(isnan(c)) > 0
    y0 = 0;
    x0 = a(1);
    x1 = b(1);
    m = ones(3,3);
else
    m = [ab; c / norm(c); cross(c / norm(c), ab)];
    
    ap = m*a';
    bp = m*b';
    
    x0 = ap(1);
    x1 = bp(1);
    y0 = ap(2);
end
end

function optDen = linearOpticalDensityOfRay(a,b, minHeight)
[x0, x1, y0, ~] = simplifyRay(a, b); %3D to 2D

if (x1 > 0 && y0 < minHeight) || (sum(~isreal([x0 x1 y0])) > 0)
    optDen = Inf;
    return;
end

if sum(~isreal([x0 x1 y0])) > 0
    optDen = 0;
    return;
end

optDen = getOpticalDepth2D(x0, x1, y0) * 10^9; %Working on nm.

end

function intensities  = atmReflectedLightOnRayForWavelengths(origin,rayDirs, ws, wis)
global nPointsPerRay;

%Computing coefficients for all rays
odps = zeros(size(rayDirs,1), size(rayDirs,2), nPointsPerRay);
odppcs = odps;
odppvs = odps;
xss = odps;
for i = 1:size(rayDirs,1)
    for j = 1:size(rayDirs,2)
        rayDir = reshape(rayDirs(i,j,:), [1,3]);
        
        [odp, odppc, odppv, xs]  = atmLuminosityCoefficientsOnRay(origin,rayDir);
        odps(i,j,:) = odp;
        odppcs(i,j,:) = odppc;
        odppvs(i,j,:) = odppv;
        xss(i,j,:) = xs;
    end
end

odppcs(isinf(odppcs)) = NaN; %Accounting for Earths shadow
odppvs(isinf(odppvs)) = NaN; %Accounting for Earths shadow

if 0% Showing atm. illumination components
    figure;
    subplot(2,2,1);
    imshow(mat2gray(mean(odps,3)));
    title('Mean Punctual Optical Depth');
    subplot(2,2,2);
    imshow(mat2gray(mean(odppcs,3)));
    title('Mean Optical Depth From Ray to Sun');
    subplot(2,2,3);
    imshow(mat2gray(mean(odppvs,3)));
    title('Optical Depth From Ray To Camera');
    subplot(2,2,4);
    imshow(mat2gray(xss(:,:,end)-xss(:,:,1)));
    title('Ray length');
end

%Calculating K
n = 1.000278; %Air refraction index http://refractiveindex.info/?shelf=other&book=air&page=Ciddor
ns = 2.545e25 * 10^-9; %Molecular air density https://en.wikipedia.org/wiki/Number_density
K = 2*pi^2*(n^2-1)^2 / (3*ns);

%K  = 7.9942e-32;

%Calculating final coeffcients
coef = 4*pi*K./((ws*10^-9).^4);

%Matrix 4D! X x Y x Point x WL
attFactor = zeros(size(rayDirs,1), size(rayDirs,2), nPointsPerRay, length(ws));
for i = 1:length(ws)
    wlAttFactor = coef(i) .* (-odppcs - odppvs); %3D
    attFactor(:,:,:,i) = wlAttFactor;
end

%We must find a K that gives us feasible values for odppcs and odppvs
%Scaling whole attFactor matrix so e^attFactor is in 0..6
K = max(max(max(max(-attFactor))))/6; %maxAttFactor should be similar to K
%K = 53328524563353944.0000; %Fixed K
% fprintf('Using K = %0.4f\n', K);
attFactor = attFactor/K;
coef = coef * K;


attFactor = exp(attFactor); %Attenuation factors appliying beer's law

intensities = zeros(size(rayDirs,1), size(rayDirs,2), length(ws));


for k = 1:length(ws)
    for i = 1:size(rayDirs,1)
        for j = 1:size(rayDirs,2)
            att = reshape(attFactor(i,j,:,k),[1,nPointsPerRay]);
            if (sum(isnan(att)) > 0)
                intensities(i,j,k) = 0;
            else
                odp = reshape(odps(i,j,:),[1,nPointsPerRay]);
                xs = reshape(xss(i,j,:),[1,nPointsPerRay]);
                
                %Simple model (Integral in nanometers)
                %intensities(i,j,k) = wis(k) .* coef(k) .* trapz(xs, odp) * 10^9;
                %Full model (Integral in nanometers)
                intensities(i,j,k) = wis(k) .* coef(k) .* trapz(xs,odp .* att) * 10^9;
            end
        end
    end
end

end

function [x0, x1] = atmIntersection2D(x0, x1, y0)
global er;
global sh;
w = sh^2-y0^2;
if w < 0
    x0 = NaN;
    x1 = NaN;
    return;
end

xa = sqrt(w);
x0 = max([-xa x0]);%Just atm.
x1 = min([x1 xa]);

q = er^2-y0^2;
if q > 0 %Touches earth
    x1 = -sqrt(q);
end

end

function [odp, odppc, odppv, xs]  = atmLuminosityCoefficientsOnRay(origin,rayDir)
global nPointsPerRay;
global er;
global sh;

[x0, x1, y0, m] = simplifyRay(origin, origin+rayDir); %3D to 2D
[x0, x1] = atmIntersection2D(x0, x1, y0); %Intersection points
if isnan(x0) || isnan(x1) || isnan(y0)
    odp = zeros(1,nPointsPerRay);
    odppc = odp;
    odppv = odp;
    xs = odp;
    return;
end

xs = sampledPoints2D(x0, x1, nPointsPerRay);

odp = zeros(1,length(xs));
odppc = odp;
odppv = odp;

for i = 1:length(xs)
    %Atm. Density
    p =[xs(i), y0];
    odp(i) = opticalDensityOfPoint(p);
    %P-Pc Scattering (from P to the Sun [assuming Sun in [0 Inf 0])
    p3D = (m\[xs(i) y0 0]')';
    
    pcx = -sh;
    pc = [pcx p3D(2) p3D(3)];
    
    odppc(i) = linearOpticalDensityOfRay(pc,p3D,er);
    %P-Pv Scattering (from P to the Viewer's position)
    odppv(i) = getOpticalDepth2D(x0, xs(i), y0) * 10^9; %Working on nm.
end

end

function xs = sampledPoints2D(x0, x1, n)

if x0*x1 < 0
    xs1 = sampledPoints2D(x0, 0, ceil(n/2));
    xs2 = sampledPoints2D(0, x1, ceil(n/2));
    xs = [xs1 xs2(2:end)];
else
    xs = (exp(-linspace(0, 4, n))-exp(-4))*(1/(1-exp(-4)));
    xs = x1 - xs *(x1-x0);
end

end

function dist = distancePointToLine(p, o, d)
b = p - (o+d);
dist = norm(cross(d,b)) / norm(d);
end

function RGB = waveLengthToRGB(w, wDelta)
%Ref: http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
XYZ = integrateWavelengthToCIE1931(w -wDelta/2, w + wDelta/2); 

m = [3.2404542 -1.5371385 -0.4985314;
    -0.9692660  1.8760108  0.0415560;
    0.0556434 -0.2040259  1.0572252];

RGB = m*XYZ';
end

function XYZ = integrateWavelengthToCIE1931(w1, w2) %W. in meters

n = 100;
ws = linspace(w1, w2, n);
XYZs = zeros(3,n);

for i = 1:n
    XYZs(:,i) = waveLengthToCIE1931(ws(i)); %W in nm.
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

%Optical Depth 2D function
function od = getOpticalDepth2D(x0, x1, y0)
global G;

if isempty(G)
    createODMap(); %Creating Map
end

if x1-x0 < 100
    od = 0;
    return;
end

[lambda0, h0] = cart2pol(x0,y0);
[lambda1, h1] = cart2pol(x1,y0);

od0 = g(lambda0, h0);
od1 = g(lambda1, h1);
od = od1 - od0;

if isnan(od) || od < 0
    figure;
    polar([lambda0, lambda1], [h0 h1], 'r');
end
end

function od = g(lambda, h)
global er;
global sh;
global G;

h = max([min([h,sh]), er]);
lambda = wrapTo2Pi(lambda);

if lambda > pi
    lambda = (2*pi) - lambda;
end

if lambda > pi/2
    %2 Cuadrant
    od = G(h, lambda);
else
    %1? Cuadrant [g(\pi/2, h) - g(\pi - \varphi, h)]
    od = 2*G(h, pi/2) - G(h, pi - lambda);
end

end

function [odmap, rmap, amap] = createODMap()
global er;
global sh;
global G;

disp(linspace(er,sh,30));
disp(linspace(pi/2,pi,30));
[rmap,amap]=meshgrid(linspace(er,sh,30),linspace(pi/2,pi,30)); %Una fila por ang

odmap = rmap;
for i=1:size(rmap,1)
    for j=1:size(rmap,2)
        x = rmap(i,j)*cos(amap(i,j));
        y = rmap(i,j)*sin(amap(i,j));
        %odmap(i,j) = cartesianDensityIntegral(x0,y0,er,sh);
        odmap(i,j) = opticalDepthOfRayFromInfinity(x,y);
    end
end

G = scatteredInterpolant(rmap(:), amap(:), odmap(:), 'linear');

if 0%Show Optical Depth Map
    figure;
    surf(rmap, amap, odmap);
    
    figure;
    rmap2 = 1+(rmap - er) / (sh-er);
    [x, y] = pol2cart(amap(:), rmap2(:));
    z = odmap(:);
    scatter3(x,y,z,100,z, 'filled');
end

end

function optDen = opticalDensityOfPoint(p)
global er;
global atmThickness;
h = max([0 (norm(p) - er)]);%Heigths
h0 = (7994 * 5e4) / atmThickness;
optDen = exp(-( h / h0)); %Density coefficient
end

function od = opticalDepthOfRayFromInfinity(x,y)
global sh;
n = 1000;
xs = linspace(-sh, x ,n);
odp = zeros(1,length(xs));
for i = 1:length(xs)
    p =[xs(i), y];
    odp(i) = opticalDensityOfPoint(p);
end
od = -trapz(abs(xs),odp) * 10^9; %Working on nm.
end