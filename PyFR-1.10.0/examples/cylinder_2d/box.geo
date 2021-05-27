d1 = 0.5;
d2 = 0.8;
d3 = 2.0;
d10 = 2.0;

Point(2) = {-200, -200, 0, 1.0};
Point(3) = {-200, 200, 0, 1.0};
Point(4) = {200, 200, 0, 1.0};
Point(5) = {200, -200, 0, 1.0};
Point(6) = {-1, 1, 0, 1.0};
Point(7) = {-1, -1, 0, 1.0};
Point(8) = {0,  d1, 0, 1.0};
Point(9) = {0, -d1, 0, 1.0};
Point(10) = {d3, 1.2, 0, 1.0};
Point(11) = {d3, -1.2, 0, 1.0};
Point(13) = {-d1, 0, 0, 1.0};
Point(14) = {-d2, 0, 0, 1.0};
Point(15) = {0, 0, 0, 1.0};
Point(16) = {d1, 0, 0, 1.0};
Point(17) = {d2, 0, 0, 1.0};
Point(18) = {0, d2, 0, 1.0};
Point(19) = {0, -d2, 0, 1.0};
Point(20) = {2*d3, 1.2, 0, 1.0};
Point(21) = {2*d3, -1.2, 0, 1.0};

Line(5) = {6, 7};
Line(6) = {7, 11};
Line(7) = {11, 10};
Line(8) = {10, 6};
Line(9) = {3, 2};
Line(10) = {2, 5};
Line(11) = {5, 4};
Line(12) = {4, 3};
//Line(13) = {16, 17};
//Line(14) = {13, 14};
//Line(15) = {8, 18};
//Line(16) = {9, 19};
//Circle(17) = {16, 15, 8};
//Circle(18) = {8, 15, 13};
//Circle(19) = {13, 15, 9};
//Circle(20) = {9, 15, 16};
//Circle(21) = {19, 15, 17};
//ircle(22) = {17, 15, 18};
//Circle(23) = {18, 15, 14};
//Circle(24) = {14, 15, 19};
Line(284) = {10, 20};
Line(285) = {20, 21};
Line(286) = {21, 11};

Line Loop(25) = {12, 9, 10, 11};
Line Loop(26) = {8, 5,6,-286, -285, -284};
Plane Surface(27) = {25, 26};
//Line Loop(28) = {22, 23, 24, 21};
//Plane Surface(29) = {26, 28};
//Line Loop(30) = {16, 21, -13, -20};
//Plane Surface(31) = {30};
//Line Loop(32) = {22, -15, -17, 13};
//Plane Surface(33) = {32};
//Line Loop(34) = {15, 23, -14, -18};
//Plane Surface(35) = {34};
//Line Loop(36) = {14, 24, -16, -19};
//Plane Surface(37) = {36};

n1 = 15;
r1 = 1.25;
n2 = 27;
n3 = 51;
n4 = 11;
n5 = 21;

//Transfinite Line {13, 15, 14, 16} = n1 Using Progression r1;
//Transfinite Line {22, 17, 23, 18, 24, 19, 20, 21} = n2 Using Progression 1;
Transfinite Line {8, 6} = n3 Using Progression 1;
Transfinite Line {5} = 21 Using Progression 1;
Transfinite Line {11, 12, 9, 10} = n5 Using Progression 1;
Transfinite Line {284, 286} = 21 Using Progression 1;
Transfinite Line {285} = 21 Using Progression 1;
Transfinite Line {7} = 21 Using Progression 1;
//Transfinite Surface {33};
//Transfinite Surface {31};
//Transfinite Surface {37};
//Transfinite Surface {35};
//Recombine Surface {33};
//Recombine Surface {35};
//Recombine Surface {37};
//Recombine Surface {31};
//Recombine Surface {29};
Recombine Surface {27};
//+
Line Loop(27) = {8, 5, 6, 7};
//+
Plane Surface(28) = {27};
//+
Line Loop(28) = {284, 285, 286, 7};
//+
Plane Surface(29) = {-28};
Transfinite Surface {28};
Transfinite Surface {29};
Recombine Surface {28};
Recombine Surface {29};

Extrude {0, 0, 0.01} {
  Surface{28,29,27}; Layers{1};Recombine;
}

Periodic Surface{308} = {28} Translate {0,0,0.01};
Periodic Surface{330} = {29} Translate {0,0,0.01};
Periodic Surface{382} = {27} Translate {0,0,0.01};

Physical Surface("FARFIELD") = {357,345,349,353};
Physical Surface("PERIODIC_2_l") = {308,330,382};
Physical Surface("PERIODIC_2_r") = {28,29,27};
Physical Volume("FLUID") = {1,2,3};

Mesh.ElementOrder = 2;
