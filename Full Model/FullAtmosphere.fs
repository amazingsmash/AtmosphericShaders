//Full Atmospheric Rayleight Scattering [Fragment Shader]
//Code by Jose Miguel Santana

//Primary view ray parameters
uniform highp vec3 uCameraPosition;
varying highp vec3 rayDir;

//Earth, atmosphere and sun parameters
const highp float earthRadius = 6.36744e6;
const highp float atmUndergroundOffset = 100e3;
const highp float atmosphereScale = 3.0; //Atm. scale
const highp float stratoHeight = 50e3 * atmosphereScale;
const highp float atmRadius = earthRadius + stratoHeight;
const highp float earth2AtmRatio = earthRadius / atmRadius;
uniform highp vec3 currentSunDir;

//Color space conversion
const highp mat3 CIE2RGB = mat3(3.2405,   -0.9693,    0.0556,
                                -1.5371,    1.8760,   -0.2040,
                                -0.4985,    0.0416,    1.0572);

const highp mat3 RGB2CIE = mat3(0.4125,    0.2127,    0.0193,
                                0.3576,    0.7152,    0.1192,
                                0.1804,    0.0722,    0.9503);

const float NaN = sqrt(-1.0);


highp vec2 raySphereIntersect(highp vec3 r0, highp vec3 rd, highp vec3 s0, highp float sr) {
  float a = dot(rd, rd);
  vec3 s0_r0 = r0 - s0;
  float b = 2.0 * dot(rd, s0_r0);
  float c = dot(s0_r0, s0_r0) - (sr * sr);
  
  float sq = b*b - 4.0*a*c;
  
  if (sq < 0.0) {
    return vec2(-1.0, -1.0);
  }
  
  sq = sqrt(b*b - 4.0*a*c);
  float s1 = (-b - sq)/(2.0*a);
  float s2 = (-b + sq)/(2.0*a);
  
  return vec2(min(s1,s2), max(s1,s2));
}

vec2 earthShadow2D(highp float Y0, highp mat3 m, vec3 sunDir){
  sunDir = normalize(sunDir);
  vec3 p2 = normalize(m * sunDir); //Sun Pos
  
  highp float x2 = p2.x;
  highp float y2 = p2.y;
  highp float z2 = p2.z;
  
  highp float R = earth2AtmRatio;
  
  highp float s = -(pow(x2,2.0)*pow(Y0,2.0)*pow(z2,2.0)) +
  pow(R,2.0)*(pow(y2,2.0) + pow(z2,2.0)) -
  (pow(y2,2.0) + pow(z2,2.0))*pow(Y0*z2,2.0);
  if (s >= 0.0){
    highp float n = x2*Y0*y2;
    highp float d = (pow(y2,2.0) + pow(z2,2.0));
    s = sqrt(s);
    highp float s1 = (n - s)/ d;
    highp float s2 = (n + s)/ d;
    
    //If the entry point is in the same direction than source
    if (distance(vec2(s1,Y0), p2.xy) <  distance(vec2(s1,Y0), -p2.xy)){
      return vec2(s1,s2);
    }
  }
  
  return vec2(NaN, NaN);
}

highp mat3 rayTo2D(in highp vec3 pa, in  highp vec3 pb){
  
  highp mat3 m;
  if (abs(pa.z) + distance(pa.y, pb.y) < 1e-7){ //Checking
    m = mat3(1.,0.,0.,
             0.,1.,0.,
             0.,0.,1.);
  } else{
    highp vec3 d = normalize(pb - pa);
    highp vec3 c = normalize(pa + (dot(-pa,d)*d));
    
    //Change of basis to 2D formula
    m[0] = d;
    m[1] = c;
    m[2] = cross(c,d);
    
    m = transpose(m);
  }
  
  return m;
}

const int nSamplesPrimaryRay = 21;

float sampledXFor21Samples(float x0, float x1, int n){
  
  if (x0*x1 < 0.0){
    if (n == 0) return x0 * 1.000000;
    if (n == 1) return x0 * 0.899549;
    if (n == 2) return x0 * 0.799199;
    if (n == 3) return x0 * 0.698949;
    if (n == 4) return x0 * 0.598800;
    if (n == 5) return x0 * 0.498750;
    if (n == 6) return x0 * 0.398800;
    if (n == 7) return x0 * 0.298951;
    if (n == 8) return x0 * 0.199201;
    if (n == 9) return x0 * 0.099551;
    
    if (n == 10) return 0.0;
    if (n == 0) return x1 * 0.099551;
    if (n == 1) return x1 * 0.199201;
    if (n == 2) return x1 * 0.298951;
    if (n == 3) return x1 * 0.398800;
    if (n == 4) return x1 * 0.498750;
    if (n == 5) return x1 * 0.598800;
    if (n == 6) return x1 * 0.698949;
    if (n == 7) return x1 * 0.799199;
    if (n == 8) return x1 * 0.899549;
    if (n == 9) return x1 * 1.000000;
  }
  
  highp float d = x1-x0;
  if (n == 0) return x0 + d * 0.000000;
  if (n == 1) return x0 + d * 0.184651;
  if (n == 2) return x0 + d * 0.335831;
  if (n == 3) return x0 + d * 0.459606;
  if (n == 4) return x0 + d * 0.560945;
  if (n == 5) return x0 + d * 0.643914;
  if (n == 6) return x0 + d * 0.711844;
  if (n == 7) return x0 + d * 0.767460;
  if (n == 8) return x0 + d * 0.812994;
  if (n == 9) return x0 + d * 0.850274;
  if (n == 10) return x0 + d * 0.880797;
  if (n == 11) return x0 + d * 0.905787;
  if (n == 12) return x0 + d * 0.926247;
  if (n == 13) return x0 + d * 0.942998;
  if (n == 14) return x0 + d * 0.956713;
  if (n == 15) return x0 + d * 0.967941;
  if (n == 16) return x0 + d * 0.977135;
  if (n == 17) return x0 + d * 0.984661;
  if (n == 18) return x0 + d * 0.990824;
  if (n == 19) return x0 + d * 0.995869;
  if (n == 20) return x0 + d * 1.000000;
}

float getDensityCoefficient(float x, float y){
  vec2 p = vec2(x, y);
  float h = (length(p) - earth2AtmRatio) / ((atmRadius - earthRadius) / atmRadius);
  float d = exp(-h*6.2547); //Density Coefficient
  return d;
}

float gFun(highp float a, highp float r){
  
  if (a > 1.6) return 4.2214e-04; //Mean value beyond a = 1.6
  r = (r - earth2AtmRatio) / (1.0 - earth2AtmRatio); //Normalizing [0..1]
  
  highp float x = r;
  highp float y = a;
  highp float x2 = x*x;
  highp float y2 = y*y;
  highp float x3 = x*x*x;
  highp float y3 = y*y*y;
  highp float x4 = x*x*x*x;
  highp float y4 = y*y*y*y;
  highp float x5 = x*x*x*x*x;
  highp float y5 = y*y*y*y*y;
  
  //Polynomial approximation
  highp float g = 1787776115.6965 + -51039518.5128*x + -5614573368.3089*y +
  865443.705*x2 + 128015133.8374*x*y + 7053069042.9223*y2 +
  -10398.8322*x3 + -1624280.2711*x2*y + -120405648.8163*x*y2 +
  -4430036061.6336*y3 + 93.8694*x4 + 12958.7149*x3*y + 1016173.9827*x2*y2 +
  50332603.1065*x*y3 + 1391246665.5755*y4 + -0.64185*x5 + -57.8523*x4*y +
  -4037.6155*x3*y2 + -211914.0726*x2*y3 + -7890102.7191*x*y4 + -174766723.1861*y5;
  
  g = clamp(g, 0.0, 1.0);
  
  return g;
}

float getOpticalDepthFromInfinity(vec2 p){
  
  highp float a = abs(atan(p.y, p.x)); // Atan->[−Pi,Pi]
  highp float r = length(p);
  
  highp float f;
  if (a > 0.0 && a < 1.57){ //0 < Pi/2
    f = 2.0 * gFun(1.5708, abs(p.y)) - gFun(3.1416 - a, r);
    f = clamp(f, 0.0, 2.0);
  } else{
    f = gFun(a, r);
  }
  
  return f;
}

float getSolarOutScattering(vec3 p, vec3 sunDir){
  //Assuming the sun is very far
  highp vec3 pc = p - normalize(sunDir);
  
  mat3 m = rayTo2D(pc, p);
  pc = m * pc;
  p = m * p;
  
  if (pc.x * p.x < 0.0  && pc.y < earth2AtmRatio){ //Earth shadow
    return -1.0; //Infinity
  } else{
    return getOpticalDepthFromInfinity(p.xy);
  }
}

float getInnerOutScattering(float x0, float x1, float y0){
  highp float i = getOpticalDepthFromInfinity(vec2(x0, y0));
  highp float f = getOpticalDepthFromInfinity(vec2(x1, y0));
  return (f - i);
}

vec4 getEarthColor(vec3 p, vec2 fragCoord){
  //Substitute with terrain rendering algorithm
  return vec4(0.,0.5, 0., 1.0);
}

vec4 getBackgroundColor(vec3 camPos, vec3 rayDir, vec3 currentSunDir, vec2 fragCoord){
  
  //Computing ray limits
  highp vec2 earthInt = raySphereIntersect(camPos, rayDir, vec3(.0,.0,.0), earthRadius);
  
  //Computing if we should put as background the Earth or the Sun
  vec4 bgColor = vec4(0.0, 0.0, 0.0, 1.0); //Space
  if (earthInt.x > 0.0){
    bgColor = getEarthColor(camPos + earthInt.x * rayDir, fragCoord);
  } else{
    vec3 sunPos = -149.6e12 * currentSunDir;
    vec2 sunInt = raySphereIntersect(camPos, rayDir, sunPos, 695.700e3);
    if (sunInt.x > 0.0){
      bgColor = vec4(1.0, 1.0, .6, 1.0);
    }
  }
  return bgColor;
}

//Output: [atm1, atm2, 0, 0] || [atm1, earth1, 0, 0] ||
// [atm1, shadow1, 0, 0] || [atm1, shadow1, shadow2, atm2] ||
// [0,0,0,0]
vec4 getRay2DEnds(vec2 p1, vec2 p2,
                  mat3 m, vec3 sunDir,
                  out bool atm,
                  out bool earth,
                  out bool shadow){
  
  //Intersection atm.
  highp float atm1 = -sqrt(1.0 - p1.y*p1.y);
  highp float atm2 = -atm1;
  atm1 = clamp(atm1, p1.x, p2.x);
  atm2 = clamp(atm2, p1.x, p2.x);
  atm = (atm1 != atm2);
  
  //Intersection earth
  highp float earth1 = -sqrt(earth2AtmRatio*earth2AtmRatio - p1.y*p1.y);
  highp float earth2 = -earth1;
  earth1 = clamp(earth1, p1.x, p2.x);
  earth2 = clamp(earth2, p1.x, p2.x);
  earth = (earth1 != earth2);
  
  //Intersection with earth shadow
  vec2 shadows = earthShadow2D(p1.y, m, sunDir);
  highp float rayEnd = earth? earth1 : atm2;
  highp float shadow1 = clamp(shadows.x, atm1, rayEnd);
  highp float shadow2 = clamp(shadows.y, atm1, rayEnd);
  
  shadow = (shadows.x == shadows.x) && (shadow1 != shadow2);
  
  //Cases:
  if (!atm){
    return vec4(0.,0.,0.,0.);
  }
  
  if (!earth && !shadow){
    return vec4(atm1, atm2, 0., 0.);
  }
  
  if (earth && !shadow){
    return vec4(atm1, earth1, 0., 0.);
  }
  
  if (earth && shadow){
    return vec4(atm1, shadow1, 0., 0.);
  }
  
  if (!earth && shadow){
    return vec4(atm1, shadow1, shadow2, atm2);
  }
}

float Xs[nSamplesPrimaryRay];
float scatteringFactors[nSamplesPrimaryRay];
float outScatteringFactors[nSamplesPrimaryRay];
void calculateScatteringFactorsAndXs(vec3 sunDir,
                                     vec3 sp1,
                                     vec3 sp2,
                                     highp float x0,
                                     highp float x1,
                                     highp float y0){
  
  highp vec3 ray3DDir = normalize(sp2-sp1);
  for (int i = 0; i < nSamplesPrimaryRay; i++){
    
    Xs[i] = sampledXFor21Samples(x0, x1, i);
    
    highp vec3 p = (Xs[i] - x0) * ray3DDir + sp1;
    
    highp float so = getSolarOutScattering(p ,sunDir);
    if (so < 0.0){
      outScatteringFactors[i] = -1.0;
    } else{
      highp float io = getInnerOutScattering(x0, Xs[i], y0);
      outScatteringFactors[i] = (so + io);
    }
    
    scatteringFactors[i] = getDensityCoefficient(Xs[i], y0);
  }
}


float getAtmWavelengthIntensity(highp float gScale4PiKw4){
  
  //Integrating
  highp float intensity = 0.0;
  
  highp float d_1 = scatteringFactors[0] * exp(- gScale4PiKw4 * outScatteringFactors[0]);
  for (int i = 1; i < nSamplesPrimaryRay; i++){
    
    highp float d = 0.0;
    if (outScatteringFactors[i] >= 0.0){
      d = scatteringFactors[i] * exp(- gScale4PiKw4 * outScatteringFactors[i]);
    }
    
    intensity += ((d + d_1)/2.0) * (Xs[i] - Xs[i-1]); //Trapezoidal integration
    
    d_1 = d; //For next iteration
  }
  
  return intensity;
}


void main() {
  //Initial points at normal mini-scale
  highp vec3 p1 = uCameraPosition/atmRadius;
  highp vec3 p2 = (uCameraPosition + normalize(rayDir)*1e10)/atmRadius;
  highp mat3 m = rayTo2D(p1, p2);
  
  highp vec2 mp1 = (m * p1).xy;
  highp vec2 mp2 = (m * p2).xy;
  if (mp1.x > mp2.x){
    highp vec2 aux = mp1;
    mp1 = mp2;
    mp2 = aux;
    highp vec3 aux3 = p1;
    p1 = p2;
    p2 = aux3;
  }
  
  bool atm;
  bool earth;
  bool shadow;
  vec4 ends = getRay2DEnds(mp1, mp2,
                           m, currentSunDir,
                           atm,
                           earth,
                           shadow);
  
  vec4 bgColor = getBackgroundColor(uCameraPosition, rayDir, currentSunDir, fragCoord);
  
  if (atm){
    
    //Auto-generated code
    const int nW = 9;
    highp vec3 CIElevels[9];
    CIElevels[0] = vec3(4.568698e-02, 2.536497e-04, 2.536274e-01); //w = 380.00 - 424.44
    CIElevels[1] = vec3(1.904479e-01, 1.411401e-02, 1.000000e+00); //w = 424.44 - 468.89
    CIElevels[2] = vec3(2.882708e-02, 1.250434e-01, 2.311921e-01); //w = 468.89 - 513.33
    CIElevels[3] = vec3(7.525619e-02, 2.688203e-01, 4.168597e-03); //w = 513.33 - 557.78
    CIElevels[4] = vec3(1.968837e-01, 1.856631e-01, 8.178416e-06); //w = 557.78 - 602.22
    CIElevels[5] = vec3(1.115830e-01, 5.053781e-02, 2.669158e-09); //w = 602.22 - 646.67
    CIElevels[6] = vec3(1.355325e-02, 6.385989e-03, 2.110279e-13); //w = 646.67 - 691.11
    CIElevels[7] = vec3(3.337662e-04, 4.290635e-04, 5.467493e-18); //w = 691.11 - 735.56
    CIElevels[8] = vec3(1.571319e-06, 1.715437e-05, 5.889087e-23); //w = 735.56 - 780.00

    highp float I0CIEKw4Scale = 3.654184e+00;
    highp float gScale4PiKw4[9];
    gScale4PiKw4[0] = 333.162685;
    gScale4PiKw4[1] = 219.071773;
    gScale4PiKw4[2] = 149.899974;
    gScale4PiKw4[3] = 105.999174;
    gScale4PiKw4[4] = 77.056384;
    gScale4PiKw4[5] = 57.351578;
    gScale4PiKw4[6] = 43.561685;
    gScale4PiKw4[7] = 33.678328;
    gScale4PiKw4[8] = 26.445582;
    //////////////////////////////////////////////
    
    vec3 atmCIE = vec3(.0,.0,.0);
    vec3 ini, end;
    //Main ray
    if (ends.x - ends.y > 1e-2){
      ini = p1 + ((ends.x - mp1.x) / (mp2.x - mp1.x)) * (p2-p1);
      end = p1 + ((ends.y - mp1.x) / (mp2.x - mp1.x)) * (p2-p1);
      
      
      //Precalculating necessary data
      calculateScatteringFactorsAndXs(currentSunDir,
                                      ini, end,
                                      ends.x, ends.y, mp1.y);
      
      for (int i = 0; i < nW; i++){
        
        highp float intensity = getAtmWavelengthIntensity(gScale4PiKw4[i]); //gScale * 4 * Pi * K / w^-4
        
        atmCIE += CIElevels[i] * intensity;
      }
    }
    
    //Secondary ray
    if (ends.w - ends.z > 1e-2){
      //Precalculating necessary data
      ini = p1 + ((ends.z - mp1.x) / (mp2.x - mp1.x)) * (p2-p1);
      end = p1 + ((ends.w - mp1.x) / (mp2.x - mp1.x)) * (p2-p1);
      
      calculateScatteringFactorsAndXs(currentSunDir,
                                      ini, end,
                                      ends.z, ends.w, mp1.y);
      
      for (int i = 0; i < nW; i++){
        
        highp float intensity = getAtmWavelengthIntensity(gScale4PiKw4[i]); //gScale * 4 * Pi * K / w^-4
        
        atmCIE += CIElevels[i] * intensity;
      }
    }
  
    //Scattering directionality
    vec3 rayDir = p2 - p1;
    float cosAngle = dot(normalize(currentSunDir), normalize(rayDir));
    float fr = (3.0/4.0)* (1.0+pow(cosAngle,2.0));
    
    atmCIE *= I0KW4CIEScale * atmRadius * fr;
    
    highp float dimmingFactor = 1e-6;
    atmCIE *= dimmingFactor;
    
    vec3 cieBG = RGB2CIE * bgColor.rgb; //Back color to CIE
    
    //Combining both colors and conversting to RGB
    vec3 atmColor = CIE2RGB * (atmCIE + bgColor);
    
    bgColor = vec4(atmColor, 1.0);
  } else{
    fragColor = bgColor;
  }

}
