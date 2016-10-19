//
//  Default.vsh
//
//  Created by José Miguel Santana Núñez: amazingsmash@gmail.com, josemiguel.santana@ulpgc.es 
//

uniform highp vec3 uCameraPosition;

varying highp vec3 planePos;

//ATM parameters
const highp float earthRadius = 6.36744e6;

const highp float atmosphereScale = 15.0;
const highp float stratoHeight = 50e3 * atmosphereScale;
const highp float atmUndergroundOffset = 100e3;

//Height at which the effect is replaced by a blue background
const highp float minHeigth = 35000.0;

//Multicolor gradient
highp vec4 whiteSky = vec4(1.0, 1.0, 1.0, 1.0);
highp vec4 blueSky = vec4(32.0 / 256.0, 173.0 / 256.0, 249.0 / 256.0, 1.0);
highp vec4 darkSpace = vec4(0.0, 0.0, 0.0, 0.0);
highp vec4 groundSkyColor = mix(blueSky, whiteSky, smoothstep(0.0, 1.0, 0.5));


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
  
  p1 = o + d * t.x;
  p2 = o + d * t.y;
  
  return length(p2-p1);
  
}

highp float getRayFactor(highp vec3 o, highp vec3 d){
  
  d /= 1000.0;
  o /= 1000.0;
  highp float er = earthRadius / 1000.0;
  highp float sh = (stratoHeight + earthRadius) / 1000.0;
  
  highp float ld = dot(d,d);
  highp float pdo = dot(d,o);
  
  highp float dx = d.x;
  highp float dy = d.y;
  highp float dz = d.z;
  
  highp float ox = o.x;
  highp float oy = o.y;
  highp float oz = o.z;
  
  highp float s = (((dx*(dx + ox) + dy*(dy + oy) + dz*(dz + oz))*
    sqrt(pow(dx + ox,2.0) + pow(dy + oy,2.0) + pow(dz + oz,2.0)))/ld -
   (sqrt(pow(ox,2.0) + pow(oy,2.0) + pow(oz,2.0))*pdo)/ld - 2.*sh +
   ((pow(dz,2.0)*(pow(ox,2.0) + pow(oy,2.0)) - 2.0*dx*dz*ox*oz - 2.0*dy*oy*(dx*ox + dz*oz) +
     pow(dy,2.0)*(pow(ox,2.0) + pow(oz,2.0)) + pow(dx,2.0)*(pow(oy,2.0) + pow(oz,2.0)))*
    log(dx*(dx + ox) + dy*(dy + oy) + dz*(dz + oz) +
        sqrt(ld)*sqrt(pow(dx + ox,2.0) + pow(dy + oy,2.0) + pow(dz + oz,2.0))))/pow(ld,1.5) -
   ((pow(dz,2.0)*(pow(ox,2.0) + pow(oy,2.0)) - 2.0*dx*dz*ox*oz - 2.0*dy*oy*(dx*ox + dz*oz) +
     pow(dy,2.0)*(pow(ox,2.0) + pow(oz,2.0)) + pow(dx,2.0)*(pow(oy,2.0) + pow(oz,2.0)))*
    log(sqrt(ld)*sqrt(pow(ox,2.0) + pow(oy,2.0) + pow(oz,2.0)) + pdo))/pow(ld,1.5))/
  (2.*(er - 1.*sh));
  
  return s;
}

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

  //Calculating color
  highp float f = getRayFactor(sp1, sp2 - sp1) * 1.3;
  
  highp vec4 color = mix(darkSpace, blueSky, smoothstep(0.0, 1.0, f));
  color = mix(color, whiteSky, smoothstep(0.7, 1.0, f));
  gl_FragColor = color;

  //Calculating camera Height (for precision problems)
  //Below a certain threshold float precision is not enough for calculations
  highp float camHeight = length(uCameraPosition) - earthRadius;
  gl_FragColor = mix(gl_FragColor, groundSkyColor, smoothstep(minHeigth, minHeigth / 4.0, camHeight));
}

