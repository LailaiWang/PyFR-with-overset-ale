//+
Point(1) = {0, 0, 0, 1.0};
//+
Point(2) = {0.5, 0, 0, 1.0};
//+
Point(3) = {0, 0.5, 0, 1.0};
//+
Point(4) = {-0.5, 0, 0, 1.0};
//+
Point(5) = {0, -0.5, 0, 1.0};
//+
Circle(1) = {2, 1, 3};
//+
Circle(2) = {3, 1, 4};
//+
Circle(3) = {4, 1, 5};
//+
Circle(4) = {5, 1, 2};
//+
Dilate {{0, 0, 0}, {2, 2, 2}} {
  Duplicata { Curve{1}; Curve{2}; Curve{3}; Curve{4}; }
}
//+
Line(9) = {8, 3};
//+
Line(10) = {6, 2};
//+
Line(11) = {18, 5};
//+
Line(12) = {4, 13};
//+
Curve Loop(1) = {1, -9, -5, 10};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {9, 2, 12, -6};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {3, -11, -7, -12};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {8, 10, -4, -11};
//+
Plane Surface(4) = {4};

Transfinite Line{1,2,3,4,5,6,7,8} = 11 Using Progression 1;
Transfinite Line{9, 10, 11, 12} = 11 Using Progression 1;

Transfinite Surface{1};
Transfinite Surface{2};
Transfinite Surface{3};
Transfinite Surface{4};

Recombine Surface{1};
Recombine Surface{2};
Recombine Surface{3};
Recombine Surface{4};

Extrude {0, 0, 0.01} {
    Surface{1,2,3,4};Layers{1};Recombine;
}

Periodic Surface{ 34} = {1} Translate {0,0,0.01};
Periodic Surface{ 56} = {2} Translate {0,0,0.01};
Periodic Surface{ 78} = {3} Translate {0,0,0.01};
Periodic Surface{100} = {4} Translate {0,0,0.01};

//+
Physical Surface("OVERSET") = {55, 29, 73, 87};
//+
Physical Surface("WALL") = {47, 65, 21, 95};
//+
Physical Surface("PERIODIC_2_l") = {2, 1, 3, 4};
//+
Physical Surface("PERIODIC_2_r") = {34, 56, 78, 100};
//+
Physical Volume("FLUID") = {2, 1, 4, 3};

Mesh.ElementOrder = 2;
