/* File: _fftfitmodule.c
 * This file is auto-generated with f2py (version:1.21.6).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * Generation date: Thu Dec  8 02:21:08 2022
 * Do not edit this file directly unless you know what you are doing!!!
 */

#ifdef __cplusplus
extern "C" {
#endif

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "Python.h"
#include <stdarg.h>
#include "fortranobject.h"
#include <math.h>

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *_fftfit_error;
static PyObject *_fftfit_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
typedef struct {float r,i;} complex_float;

/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif

#define rank(var) var ## _Rank
#define shape(var,dim) var ## _Dims[dim]
#define old_rank(var) (PyArray_NDIM((PyArrayObject *)(capi_ ## var ## _tmp)))
#define old_shape(var,dim) PyArray_DIM(((PyArrayObject *)(capi_ ## var ## _tmp)),dim)
#define fshape(var,dim) shape(var,rank(var)-dim-1)
#define len(var) shape(var,0)
#define flen(var) fshape(var,0)
#define old_size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))
/* #define index(i) capi_i ## i */
#define slen(var) capi_ ## var ## _len
#define size(var, ...) f2py_size((PyArrayObject *)(capi_ ## var ## _tmp), ## __VA_ARGS__, -1)

#define CHECKSCALAR(check,tcheck,name,show,var)\
    if (!(check)) {\
        char errstring[256];\
        sprintf(errstring, "%s: "show, "("tcheck") failed for "name, var);\
        PyErr_SetString(_fftfit_error,errstring);\
        /*goto capi_fail;*/\
    } else 
#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
    fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif


/************************ See f2py2e/cfuncs.py: cfuncs ************************/
static int f2py_size(PyArrayObject* var, ...)
{
  npy_int sz = 0;
  npy_int dim;
  npy_int rank;
  va_list argp;
  va_start(argp, var);
  dim = va_arg(argp, npy_int);
  if (dim==-1)
    {
      sz = PyArray_SIZE(var);
    }
  else
    {
      rank = PyArray_NDIM(var);
      if (dim>=1 && dim<=rank)
        sz = PyArray_DIM(var, dim-1);
      else
        fprintf(stderr, "f2py_size: 2nd argument value=%d fails to satisfy 1<=value<=%d. Result will be 0.\n", dim, rank);
    }
  va_end(argp);
  return sz;
}

static int
int_from_pyobj(int* v, PyObject *obj, const char *errmess)
{
    PyObject* tmp = NULL;

    if (PyLong_Check(obj)) {
        *v = Npy__PyLong_AsInt(obj);
        return !(*v == -1 && PyErr_Occurred());
    }

    tmp = PyNumber_Long(obj);
    if (tmp) {
        *v = Npy__PyLong_AsInt(tmp);
        Py_DECREF(tmp);
        return !(*v == -1 && PyErr_Occurred());
    }

    if (PyComplex_Check(obj)) {
        PyErr_Clear();
        tmp = PyObject_GetAttrString(obj,"real");
    }
    else if (PyBytes_Check(obj) || PyUnicode_Check(obj)) {
        /*pass*/;
    }
    else if (PySequence_Check(obj)) {
        PyErr_Clear();
        tmp = PySequence_GetItem(obj, 0);
    }

    if (tmp) {
        if (int_from_pyobj(v, tmp, errmess)) {
            Py_DECREF(tmp);
            return 1;
        }
        Py_DECREF(tmp);
    }

    {
        PyObject* err = PyErr_Occurred();
        if (err == NULL) {
            err = _fftfit_error;
        }
        PyErr_SetString(err, errmess);
    }
    return 0;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
extern void F_FUNC(cprof,CPROF)(float*,int*,int*,complex_float*,float*,float*);
extern void F_FUNC(fftfit,FFTFIT)(float*,float*,float*,int*,float*,float*,float*,float*,float*,float*,int*);
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/*********************************** cprof ***********************************/
static char doc_f2py_rout__fftfit_cprof[] = "\
c,amp,pha = cprof(y,[nmax,nh])\n\nWrapper for ``cprof``.\
\n\nParameters\n----------\n"
"y : input rank-1 array('f') with bounds (nmax)\n"
"\nOther Parameters\n----------------\n"
"nmax : input int, optional\n    Default: len(y)\n"
"nh : input int, optional\n    Default: (nmax/2)\n"
"\nReturns\n-------\n"
"c : rank-1 array('F') with bounds (nh + 1)\n"
"amp : rank-1 array('f') with bounds (nh)\n"
"pha : rank-1 array('f') with bounds (nh)";
/* extern void F_FUNC(cprof,CPROF)(float*,int*,int*,complex_float*,float*,float*); */
static PyObject *f2py_rout__fftfit_cprof(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(float*,int*,int*,complex_float*,float*,float*)) {
    PyObject * volatile capi_buildvalue = NULL;
    volatile int f2py_success = 1;
/*decl*/

  float *y = NULL;
  npy_intp y_Dims[1] = {-1};
  const int y_Rank = 1;
  PyArrayObject *capi_y_tmp = NULL;
  int capi_y_intent = 0;
  PyObject *y_capi = Py_None;
  int nmax = 0;
  PyObject *nmax_capi = Py_None;
  int nh = 0;
  PyObject *nh_capi = Py_None;
  complex_float *c = NULL;
  npy_intp c_Dims[1] = {-1};
  const int c_Rank = 1;
  PyArrayObject *capi_c_tmp = NULL;
  int capi_c_intent = 0;
  float *amp = NULL;
  npy_intp amp_Dims[1] = {-1};
  const int amp_Rank = 1;
  PyArrayObject *capi_amp_tmp = NULL;
  int capi_amp_intent = 0;
  float *pha = NULL;
  npy_intp pha_Dims[1] = {-1};
  const int pha_Rank = 1;
  PyArrayObject *capi_pha_tmp = NULL;
  int capi_pha_intent = 0;
    static char *capi_kwlist[] = {"y","nmax","nh",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
    if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
        "O|OO:_fftfit.cprof",\
        capi_kwlist,&y_capi,&nmax_capi,&nh_capi))
        return NULL;
/*frompyobj*/
  /* Processing variable y */
  ;
  capi_y_intent |= F2PY_INTENT_IN;
  capi_y_tmp = array_from_pyobj(NPY_FLOAT,y_Dims,y_Rank,capi_y_intent,y_capi);
  if (capi_y_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _fftfit_error,"failed in converting 1st argument `y' of _fftfit.cprof to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    y = (float *)(PyArray_DATA(capi_y_tmp));

  /* Processing variable nmax */
  if (nmax_capi == Py_None) nmax = len(y); else
    f2py_success = int_from_pyobj(&nmax,nmax_capi,"_fftfit.cprof() 1st keyword (nmax) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(len(y)>=nmax,"len(y)>=nmax","1st keyword nmax","cprof:nmax=%d",nmax) {
  /* Processing variable nh */
  if (nh_capi == Py_None) nh = (nmax/2); else
    f2py_success = int_from_pyobj(&nh,nh_capi,"_fftfit.cprof() 2nd keyword (nh) can't be converted to int");
  if (f2py_success) {
  /* Processing variable c */
  c_Dims[0]=nh + 1;
  capi_c_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE;
  capi_c_tmp = array_from_pyobj(NPY_CFLOAT,c_Dims,c_Rank,capi_c_intent,Py_None);
  if (capi_c_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _fftfit_error,"failed in converting hidden `c' of _fftfit.cprof to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    c = (complex_float *)(PyArray_DATA(capi_c_tmp));

  /* Processing variable amp */
  amp_Dims[0]=nh;
  capi_amp_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE;
  capi_amp_tmp = array_from_pyobj(NPY_FLOAT,amp_Dims,amp_Rank,capi_amp_intent,Py_None);
  if (capi_amp_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _fftfit_error,"failed in converting hidden `amp' of _fftfit.cprof to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    amp = (float *)(PyArray_DATA(capi_amp_tmp));

  /* Processing variable pha */
  pha_Dims[0]=nh;
  capi_pha_intent |= F2PY_INTENT_OUT|F2PY_INTENT_HIDE;
  capi_pha_tmp = array_from_pyobj(NPY_FLOAT,pha_Dims,pha_Rank,capi_pha_intent,Py_None);
  if (capi_pha_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _fftfit_error,"failed in converting hidden `pha' of _fftfit.cprof to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    pha = (float *)(PyArray_DATA(capi_pha_tmp));

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(y,&nmax,&nh,c,amp,pha);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
        if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
        CFUNCSMESS("Building return value.\n");
        capi_buildvalue = Py_BuildValue("NNN",capi_c_tmp,capi_amp_tmp,capi_pha_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
        } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  }  /*if (capi_pha_tmp == NULL) ... else of pha*/
  /* End of cleaning variable pha */
  }  /*if (capi_amp_tmp == NULL) ... else of amp*/
  /* End of cleaning variable amp */
  }  /*if (capi_c_tmp == NULL) ... else of c*/
  /* End of cleaning variable c */
  } /*if (f2py_success) of nh*/
  /* End of cleaning variable nh */
  } /*CHECKSCALAR(len(y)>=nmax)*/
  } /*if (f2py_success) of nmax*/
  /* End of cleaning variable nmax */
  if((PyObject *)capi_y_tmp!=y_capi) {
    Py_XDECREF(capi_y_tmp); }
  }  /*if (capi_y_tmp == NULL) ... else of y*/
  /* End of cleaning variable y */
/*end of cleanupfrompyobj*/
    if (capi_buildvalue == NULL) {
/*routdebugfailure*/
    } else {
/*routdebugleave*/
    }
    CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
    return capi_buildvalue;
}
/******************************** end of cprof ********************************/

/*********************************** fftfit ***********************************/
static char doc_f2py_rout__fftfit_fftfit[] = "\
shift,eshift,snr,esnr,b,errb,ngood = fftfit(prof,s,phi,[nmax])\n\nWrapper for ``fftfit``.\
\n\nParameters\n----------\n"
"prof : input rank-1 array('f') with bounds (nmax)\n"
"s : input rank-1 array('f') with bounds (0.5 * nmax)\n"
"phi : input rank-1 array('f') with bounds (0.5 * nmax)\n"
"\nOther Parameters\n----------------\n"
"nmax : input int, optional\n    Default: len(prof)\n"
"\nReturns\n-------\n"
"shift : float\n"
"eshift : float\n"
"snr : float\n"
"esnr : float\n"
"b : float\n"
"errb : float\n"
"ngood : int";
/* extern void F_FUNC(fftfit,FFTFIT)(float*,float*,float*,int*,float*,float*,float*,float*,float*,float*,int*); */
static PyObject *f2py_rout__fftfit_fftfit(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(float*,float*,float*,int*,float*,float*,float*,float*,float*,float*,int*)) {
    PyObject * volatile capi_buildvalue = NULL;
    volatile int f2py_success = 1;
/*decl*/

  float *prof = NULL;
  npy_intp prof_Dims[1] = {-1};
  const int prof_Rank = 1;
  PyArrayObject *capi_prof_tmp = NULL;
  int capi_prof_intent = 0;
  PyObject *prof_capi = Py_None;
  float *s = NULL;
  npy_intp s_Dims[1] = {-1};
  const int s_Rank = 1;
  PyArrayObject *capi_s_tmp = NULL;
  int capi_s_intent = 0;
  PyObject *s_capi = Py_None;
  float *phi = NULL;
  npy_intp phi_Dims[1] = {-1};
  const int phi_Rank = 1;
  PyArrayObject *capi_phi_tmp = NULL;
  int capi_phi_intent = 0;
  PyObject *phi_capi = Py_None;
  int nmax = 0;
  PyObject *nmax_capi = Py_None;
  float shift = 0;
  float eshift = 0;
  float snr = 0;
  float esnr = 0;
  float b = 0;
  float errb = 0;
  int ngood = 0;
    static char *capi_kwlist[] = {"prof","s","phi","nmax",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
    if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
        "OOO|O:_fftfit.fftfit",\
        capi_kwlist,&prof_capi,&s_capi,&phi_capi,&nmax_capi))
        return NULL;
/*frompyobj*/
  /* Processing variable prof */
  ;
  capi_prof_intent |= F2PY_INTENT_IN;
  capi_prof_tmp = array_from_pyobj(NPY_FLOAT,prof_Dims,prof_Rank,capi_prof_intent,prof_capi);
  if (capi_prof_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _fftfit_error,"failed in converting 1st argument `prof' of _fftfit.fftfit to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    prof = (float *)(PyArray_DATA(capi_prof_tmp));

  /* Processing variable shift */
  /* Processing variable eshift */
  /* Processing variable snr */
  /* Processing variable esnr */
  /* Processing variable b */
  /* Processing variable errb */
  /* Processing variable ngood */
  /* Processing variable nmax */
  if (nmax_capi == Py_None) nmax = len(prof); else
    f2py_success = int_from_pyobj(&nmax,nmax_capi,"_fftfit.fftfit() 1st keyword (nmax) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(len(prof)>=nmax,"len(prof)>=nmax","1st keyword nmax","fftfit:nmax=%d",nmax) {
  /* Processing variable s */
  s_Dims[0]=0.5 * nmax;
  capi_s_intent |= F2PY_INTENT_IN;
  capi_s_tmp = array_from_pyobj(NPY_FLOAT,s_Dims,s_Rank,capi_s_intent,s_capi);
  if (capi_s_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _fftfit_error,"failed in converting 2nd argument `s' of _fftfit.fftfit to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    s = (float *)(PyArray_DATA(capi_s_tmp));

  /* Processing variable phi */
  phi_Dims[0]=0.5 * nmax;
  capi_phi_intent |= F2PY_INTENT_IN;
  capi_phi_tmp = array_from_pyobj(NPY_FLOAT,phi_Dims,phi_Rank,capi_phi_intent,phi_capi);
  if (capi_phi_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _fftfit_error,"failed in converting 3rd argument `phi' of _fftfit.fftfit to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    phi = (float *)(PyArray_DATA(capi_phi_tmp));

/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(prof,s,phi,&nmax,&shift,&eshift,&snr,&esnr,&b,&errb,&ngood);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
        if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
        CFUNCSMESS("Building return value.\n");
        capi_buildvalue = Py_BuildValue("ffffffi",shift,eshift,snr,esnr,b,errb,ngood);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
        } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  if((PyObject *)capi_phi_tmp!=phi_capi) {
    Py_XDECREF(capi_phi_tmp); }
  }  /*if (capi_phi_tmp == NULL) ... else of phi*/
  /* End of cleaning variable phi */
  if((PyObject *)capi_s_tmp!=s_capi) {
    Py_XDECREF(capi_s_tmp); }
  }  /*if (capi_s_tmp == NULL) ... else of s*/
  /* End of cleaning variable s */
  } /*CHECKSCALAR(len(prof)>=nmax)*/
  } /*if (f2py_success) of nmax*/
  /* End of cleaning variable nmax */
  /* End of cleaning variable ngood */
  /* End of cleaning variable errb */
  /* End of cleaning variable b */
  /* End of cleaning variable esnr */
  /* End of cleaning variable snr */
  /* End of cleaning variable eshift */
  /* End of cleaning variable shift */
  if((PyObject *)capi_prof_tmp!=prof_capi) {
    Py_XDECREF(capi_prof_tmp); }
  }  /*if (capi_prof_tmp == NULL) ... else of prof*/
  /* End of cleaning variable prof */
/*end of cleanupfrompyobj*/
    if (capi_buildvalue == NULL) {
/*routdebugfailure*/
    } else {
/*routdebugleave*/
    }
    CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
    return capi_buildvalue;
}
/******************************* end of fftfit *******************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/
/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {
  {"cprof",-1,{{-1}},0,(char *)F_FUNC(cprof,CPROF),(f2py_init_func)f2py_rout__fftfit_cprof,doc_f2py_rout__fftfit_cprof},
  {"fftfit",-1,{{-1}},0,(char *)F_FUNC(fftfit,FFTFIT),(f2py_init_func)f2py_rout__fftfit_fftfit,doc_f2py_rout__fftfit_fftfit},

/*eof routine_defs*/
  {NULL}
};

static PyMethodDef f2py_module_methods[] = {

  {NULL,NULL}
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_fftfit",
  NULL,
  -1,
  f2py_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyMODINIT_FUNC PyInit__fftfit(void) {
  int i;
  PyObject *m,*d, *s, *tmp;
  m = _fftfit_module = PyModule_Create(&moduledef);
  Py_SET_TYPE(&PyFortran_Type, &PyType_Type);
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module _fftfit (failed to import numpy)"); return m;}
  d = PyModule_GetDict(m);
  s = PyUnicode_FromString("1.21.6");
  PyDict_SetItemString(d, "__version__", s);
  Py_DECREF(s);
  s = PyUnicode_FromString(
    "This module '_fftfit' is auto-generated with f2py (version:1.21.6).\nFunctions:\n"
"  c,amp,pha = cprof(y,nmax=len(y),nh=(nmax/2))\n"
"  shift,eshift,snr,esnr,b,errb,ngood = fftfit(prof,s,phi,nmax=len(prof))\n"
".");
  PyDict_SetItemString(d, "__doc__", s);
  Py_DECREF(s);
  s = PyUnicode_FromString("1.21.6");
  PyDict_SetItemString(d, "__f2py_numpy_version__", s);
  Py_DECREF(s);
  _fftfit_error = PyErr_NewException ("_fftfit.error", NULL, NULL);
  /*
   * Store the error object inside the dict, so that it could get deallocated.
   * (in practice, this is a module, so it likely will not and cannot.)
   */
  PyDict_SetItemString(d, "__fftfit_error", _fftfit_error);
  Py_DECREF(_fftfit_error);
  for(i=0;f2py_routine_defs[i].name!=NULL;i++) {
    tmp = PyFortranObject_NewAsAttr(&f2py_routine_defs[i]);
    PyDict_SetItemString(d, f2py_routine_defs[i].name, tmp);
    Py_DECREF(tmp);
  }


/*eof initf2pywraphooks*/
/*eof initf90modhooks*/

/*eof initcommonhooks*/


#ifdef F2PY_REPORT_ATEXIT
  if (! PyErr_Occurred())
    on_exit(f2py_report_on_exit,(void*)"_fftfit");
#endif
  return m;
}
#ifdef __cplusplus
}
#endif