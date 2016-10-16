//
//  Default.vsh
//
//  Created by José Miguel Santana Núñez
//

uniform highp vec3 uCameraPosition;

varying highp vec3 planePos;

//ATM parameters
const highp float earthRadius = 6.36744e6;

const highp float atmosphereScale = 15.0;
const highp float tropoHeight = 10e3 * atmosphereScale;
const highp float stratoHeight = 50e3 * atmosphereScale;

//const highp float atmThickness = 50e3;
const highp float atmUndergroundOffset = 100e3;

//Max distance on atmosphere (as in: http://www.mathopenref.com/chord.html )
const highp float maxDistTropo = 2.0 * sqrt(pow(earthRadius + tropoHeight, 2.0) - pow(earthRadius, 2.0));
const highp float maxDistStrato = 2.0 * sqrt(pow(earthRadius + stratoHeight, 2.0) - pow(earthRadius + tropoHeight, 2.0));

//Multicolor gradient
highp vec4 whiteSky = vec4(1.0, 1.0, 1.0, 1.0);
highp vec4 blueSky = vec4(32.0 / 256.0, 173.0 / 256.0, 249.0 / 256.0, 1.0);//vec4(128.0 / 256.0, 128.0 / 256.0, 256.0 / 256.0, 1.0);
highp vec4 darkSpace = vec4(0.0, 0.0, 0.0, 0.0);

highp vec4 red = vec4(1.0, 0.0, 0.0, 1.0);
highp vec4 green = vec4(0.0, 1.0, 0.0, 1.0);

highp vec2 intersectionsWithSphere(highp vec3 o,
                                   highp vec3 d,
                                   highp float r){
  //http://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-sphere-intersection
  
  highp float a = dot(d,d);
  highp float b = 2.0 * dot(o,d);
  highp float c = dot(o,o) - pow(r, 2.0);
  
  highp float q = pow(b,2.0) - 4.0 * a * c;
  if (q < 0.0){
    return vec2(-1.0, -1.0); //No idea how to write NAN in GLSL
  }
  
  highp float sq = sqrt(q);
  highp float t1 = (-b - sq) / (2.0*a);
  highp float t2 = (-b + sq) / (2.0*a);
  
  if (t1 < t2){
    return vec2(t1,t2);
  } else{
    return vec2(t2, t1);
  }
}

highp float rayLenghtInSphere(highp vec3 o,
                              highp vec3 d,
                              highp float r,
                              out highp vec3 p1,
                              out highp vec3 p2){
  highp vec2 t = intersectionsWithSphere(o,d,r);
  
  if (t.x < 0.0){
    if (t.y < 0.0){
      return 0.0;
    } else{
      t.x = 0.0;
    }
  }
  
  if (t.x < 1.0){ //Eliminating distance to Znear plane
    t.x = 1.0;
  }
  
  p1 = o + d * t.x;
  p2 = o + d * t.y;
  
  return length(p2-p1);
  
}

highp float getFactor(highp float rayLength, highp float maxRayLength){
  highp float f = rayLength / maxRayLength;
  //  if (f < 0.999){
  //    f = log(1.0 - f) / -4.7;
  //  }
  //  if (f > 1.0){
  //    f = 1.0;
  //  }
  //  if (f < 0.0){
  //    f = 0.0;
  //  }
  return f;
}


highp float getRayFactor(highp vec3 p1,
                         highp vec3 p2,
                         highp float minR,
                         highp float maxR){
  
  //Height
//  highp float hp1 = length(p1);
//  highp float hp2 = length(p2);
  highp float h = length((p1 + p2) / 2.0);
//  highp float h = (hp1 + hp2 + 2.0 * hpc) / 4.0;
//  highp float h = (hp1 + hp2) / 2.0;
  
//  if (h > maxR){
//    return -1.0;
//  }
  
  highp float heightFactor = 1.0 - ((h - minR) / (maxR - minR));
  
//  if (heightFactor < 0.0){
//    return -1.0;
//  }
  return heightFactor;
  
  //Length
  highp float maxLength = 2.0 * sqrt(pow(maxR, 2.0) - pow(minR, 2.0));
  highp float rayLength = length(p2-p1);
  
  highp float lengthFactor = rayLength / maxLength;
  
  return heightFactor * lengthFactor;
}

////////////////////////

void main() {
  
  
  //Ray [O + tD = X]
  highp vec3 o = planePos;
  highp vec3 d = planePos - uCameraPosition;
  
  //Discarding pixels on Earth
  highp vec2 interEarth = intersectionsWithSphere(o,d, earthRadius - atmUndergroundOffset);
  if (interEarth.x != -1.0 || interEarth.y != -1.0){
    discard;
  }
  
  //Ray length in stratosphere
  highp vec3 sp1, sp2;
  highp float stratoLength = rayLenghtInSphere(o,d, earthRadius + stratoHeight, sp1, sp2);
  if (stratoLength <= 0.0){
    discard;
  }
  //Ray length in troposhpere
  highp vec3 tp1, tp2;
  highp float tropoLength = rayLenghtInSphere(o,d, earthRadius + tropoHeight, tp1, tp2);
  if (tropoLength > 0.0){
    stratoLength -= tropoLength;
  }
  
  //Refraction factors
  highp float stratoFactor = getFactor(stratoLength, maxDistStrato);
  highp float tropoFactor = getFactor(tropoLength, maxDistTropo);
  
//  highp float stratoFactor = 0.0;
//  if (tropoLength <= 0.0){
//    
//    stratoFactor = getRayFactor(sp1, sp2,
//                                earthRadius + tropoHeight,
//                                earthRadius + stratoHeight);
//  } else{
//    stratoFactor = getRayFactor(sp1, tp1,
//                                earthRadius + tropoHeight,
//                                earthRadius + stratoHeight);
//    stratoFactor += getRayFactor(tp2, sp2,
//                                 earthRadius + tropoHeight,
//                                 earthRadius + stratoHeight);
//  }
//  
//  
//  highp float tropoFactor = getRayFactor(tp1, tp2,
//                                         earthRadius,
//                                         earthRadius + tropoHeight);
  
  highp float f =  (tropoFactor + stratoFactor);
  
  highp vec4 color = mix(darkSpace, blueSky, smoothstep(0.0, 1.0, f));
  color = mix(color, whiteSky, smoothstep(1.0, 1.6, f));
  gl_FragColor = color;
  
  
  //Calculating camera Height (for precision problems)
  //Below a certain threshold float precision is not enough for calculations
  const highp float minHeigth = 20000.0;
  highp float camHeight = length(uCameraPosition) - earthRadius;
  gl_FragColor = mix(gl_FragColor, blueSky, smoothstep(minHeigth, minHeigth / 2.0, camHeight));
  
}

