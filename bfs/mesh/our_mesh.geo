DefineConstant
 [
   inf_width = {0.2, Name "inflow width"}
   step_height = {0.1, Name "step height"}
   step_dist = {1.0, Name "step distance"}
   top_len = {3.0, Name "top length"}
   control_width = {0.02, Name "control width"}
   inflow_size = {0.02, Name "size for inflow"}
   control_size= {0.004, Name "size control"}
   step_size = {0.016, Name "size past step"}
   outflow_size = {0.03, Name "size for outflow"}
 ];
 
 length_bef_control = step_dist - control_width;
 outflow_width = step_height + inf_width;
 bott_len = top_len - step_dist;
 
 p = newp;
 
 Point(p+1) = {0, step_height, 0, inflow_size};
 Point(p+2) = {length_bef_control, step_height, 0, control_size};
 Point(p+3) = {step_dist, step_height, 0, control_size};
 Point(p+4) = {step_dist, 0, 0, step_size};
 Point(p+5) = {top_len, 0, 0, outflow_size};
 Point(p+6) = {top_len, outflow_width, 0, outflow_size};
 Point(p+7) = {0, outflow_width, 0, inflow_size};
 
 l = newl;
 
 Line(l+1) = {p+1, p+2};
 Line(l+2) = {p+2, p+3};
 Line(l+3) = {p+3, p+4};
 Line(l+4) = {p+4, p+5};
 Line(l+5) = {p+5, p+6};
 Line(l+6) = {p+6, p+7};
 Line(l+7) = {p+7, p+1};
 
 // Outflow
 out = 2;
 Physical Line(out) = {l+5};
 
 // No slip wall
 noslip1=3;
 noslip2=4;
 noslip3=5;
 Physical Line(noslip1) = {l+1};
 Physical Line(noslip2) = {l+3, l+4};
 Physical Line(noslip3) = {l+6};
 
 // Inlet
 in=1;
 Physical Line(in) = {l+7};
 
 // control line
 control=6;
 Physical Line(control) = {l+2};
 
 bd = newll; 
 Curve Loop(bd) = {(l+1), (l+2), (l+3), (l+4), (l+5), (l+6), (l+7)}; 
 
 s = news;
 
 Plane Surface(s) = {bd};
 tot=1;
 Physical Surface(tot) = {s};
 
 // DefineConstant[
 //   inf_width = {0.2, Name "inflow widtth"}
 //   step_height = {0.1, Name "step height"}
 //   step_dist = {1.0, Name "step distance"}
 //   top_len = {3.0, Name "top length"}
 //   control_width = {0.02, Name "control wdth"}
 //   refined_size = {0.002, Name "size for step"}
 //   coarse_size= {0.01, Name "size coarse"}
 // ];
 
 // length_bef_control = step_dist - control_width;
 // outflow_width = step_height + inf_width;
 // bott_len = top_len - step_dist;
 
 // p = newp;
 
 // Point(p+0) = {0, step_height, 0, coarse_size};
 // Point(p+1) = {length_bef_control, step_height, 0, refined_size};
 // Point(p+2) = {step_dist, step_height, 0, refined_size};
 // Point(p+3) = {step_dist, 0, 0, refined_size};
 // Point(p+4) = {top_len, 0, 0, coarse_size};
 // Point(p+5) = {top_len, outflow_width, 0, coarse_size};
 // Point(p+6) = {0, outflow_width, 0, coarse_size};
 
 // l = newl;
 
 // Line(l) = {p, p+1};
 // Line(l+1) = {p+1, p+2};
 // Line(l+2) = {p+2, p+3};
 // Line(l+3) = {p+3, p+4};
 // Line(l+4) = {p+4, p+5};
 // Line(l+5) = {p+5, p+6};
 // Line(l+6) = {p+6, p};
 
 // // Outflow
 // out = 1;
 // Physical Curve(out) = {l+4};
 
 // // No slip wall
 // noslip1=2;
 // noslip2=3;
 // noslip3=4;
 // Physical Curve(noslip1) = {l};
 // Physical Curve(noslip2) = {l+2, l+3};
 // Physical Curve(noslip3) = {l+5};
 
 // // Inlet
 // in=5;
 // Physical Curve(in) = {l+6};
 
 // // control line
 // control=6;
 // Physical Curve(control) = {l+1};
 
 // bd = newll; 
 // Curve Loop(bd) = {(l), (l+1), (l+2), (l+3), (l+4), (l+5), (l+6)}; 
 
 // s = news;
 
 // Plane Surface(s) = {bd};
 // tot=1;
 // Physical Surface(tot) = {s};
 
 // //Characteristic Length{cylinder[]} = cylinder_size;
 // //Characteristic Length{coarse[]} = coarse_size;
 // //Characteristic Length{cframe[]} = box_size;