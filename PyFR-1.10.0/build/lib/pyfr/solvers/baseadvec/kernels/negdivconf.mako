# -*- coding: utf-8 -*-
<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='negdivconf' ndim='2'
              t='scalar fpdtype_t'
              tdivtconf='inout fpdtype_t[${str(nvars)}]'
              ploc='in fpdtype_t[${str(ndims)}]'
              u='in fpdtype_t[${str(nvars)}]'
              s='in fpdtype_t[${str(nvars)}]'
              rcpdjac='in fpdtype_t'
              gclcomp='in fpdtype_t'>
% for i, ex in enumerate(srcex):
    tdivtconf[${i}] = -rcpdjac*tdivtconf[${i}] + ${ex};
% endfor

// add the gcl component into RHS
% if mvgrid is True:
% for i in range(nvars):
    tdivtconf[${i}] = tdivtconf[${i}]-rcpdjac*s[${i}]*gclcomp;
% endfor
% endif
</%pyfr:kernel>
