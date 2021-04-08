# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<% one = 1 %>
<%pyfr:kernel name ='gclcomponent' ndim='2'
              smats='in    fpdtype_t[${str(ndims)}][${str(ndims)}]'
              mvel ='inout fpdtype_t[${str(ndims)}][${str(one)}]'>

    // copy data 
    fpdtype_t mveltmp[${ndims}][${one}];

% for i,j in pyfr.ndrange(ndims,one):
    mveltmp[${i}][${j}] = mvel[${i}][${j}];
% endfor

    // calculate 
    // |J|*ξ_x*mvelx + |J|*ξ_y*mvely + |J|*ξ_z*mvelz
    // |J|*η_x*mvelx + |J|*η_y*mvely + |J|*η_z*mvelz
    // |J|*ζ_x*mvelx + |J|*ζ_y*mvely + |J|*ζ_z*mvelz

% for i,j in pyfr.ndrange(ndims,one):
    mvel[${i}][${j}] = ${' + '.join('smats[{0}][{1}]*mveltmp[{1}][{2}]'
                              .format(i,k,j)
                              for k in range(ndims))};
% endfor

</%pyfr:kernel>
