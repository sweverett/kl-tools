#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <string>
#include <complex>
#include <vector>
#include <time.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

//PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<double>);

namespace py = pybind11;

#include "galsim/GSParams.h"
#include "galsim/SBInclinedExponential.h"
#include "galsim/Image.h"
#include "galsim/Random.h"

#define idx2( i, Ni, j, Nj ) ((int)( (i) * (Nj) + (j) ))
#define idx3(i, Ni, j, Nj, k, Nk) ((int)( ( (i) * (Nj) + (j) ) * (Nk) + (k) ))
#define PI (3.14159265359)
using namespace std;

/* Parameter Structure
 *
 * TODO:
 *  - wrap these parameters and related functions into a class structure.
 *    Note that since these parameters are related with specific exposure,
 *    the class object should be similar to "data_generator" type object in
 *    the kl-tools pipeline
 *  - checkout the latest Spencer's updates, and think about merge the repos.
 *    Should tail the code such that easy fit into Spencer's branch.
 * Related parameters
 * - theory cube pixel scale
 * - theory cube pixel dimension
 * - observation pixel scale
 * - observation pixel dimension
 * - bandpass file (pass as argument?)
 * - PSF (ignore chromatic PSF, pass)
 * - dispersion relation dx/dlam = R_spec / 500
 * - dispersion angle (rad)
 * - observation frame offset
 *
 * */
typedef struct {
    int theory_cube_Nx;
    int theory_cube_Ny;
    int theory_cube_Nlam;
    double theory_cube_scale;
    int observed_image_Nx;
    int observed_image_Ny;
    double observed_image_scale;
    double R_spec;
    double disper_angle;
    double offset;
    // exposure time calculation
    double diameter_in_cm;
    double exptime_in_sec;
    double gain;
} Pars;
Pars pars = {.theory_cube_Nx = 0,
             .theory_cube_Ny = 0,
             .theory_cube_Nlam = 0,
             .theory_cube_scale = 0.0,
             .observed_image_Nx = 0,
             .observed_image_Ny = 0,
             .observed_image_scale = 0.0,
             .R_spec = 0.0,
             .disper_angle = 0.0,
             .offset = 0.0,
             .diameter_in_cm = 0.0,
             .exptime_in_sec = 0.0,
             .gain = 0.0};

int cpp_set_pars(int theory_cube_Nx, int theory_cube_Ny, int theory_cube_Nlam,
                   double theory_cube_scale, int observed_image_Nx,
                   int observed_image_Ny, double observed_image_scale,
                   double R_spec, double disper_angle, double offset,
                   double diameter_in_cm, double exptime_in_sec, double gain)
{
    pars.theory_cube_Nx = theory_cube_Nx;
    pars.theory_cube_Ny = theory_cube_Ny;
    pars.theory_cube_Nlam = theory_cube_Nlam;
    pars.theory_cube_scale = theory_cube_scale;
    pars.observed_image_Nx = observed_image_Nx;
    pars.observed_image_Ny = observed_image_Ny;
    pars.observed_image_scale = observed_image_scale;
    pars.R_spec = R_spec;
    pars.disper_angle = disper_angle;
    pars.offset = offset;
    pars.diameter_in_cm = diameter_in_cm;
    pars.exptime_in_sec = exptime_in_sec;
    pars.gain = gain;

    return 0;
}

int cpp_print_Pars()
{
    cout << "Print Pars object:" << endl;
    cout << "\t - theory_cube_Nx = " << pars.theory_cube_Nx << endl;
    cout << "\t - theory_cube_Ny = " << pars.theory_cube_Ny << endl;
    cout << "\t - theory_cube_Nlam = " << pars.theory_cube_Nlam << endl;
    cout << "\t - theory_cube_scale = " << pars.theory_cube_scale << endl;
    cout << "\t - observed_image_Nx = " << pars.observed_image_Nx << endl;
    cout << "\t - observed_image_Ny = " << pars.observed_image_Ny << endl;
    cout << "\t - observed_image_scale = " << pars.observed_image_scale << endl;
    cout << "\t - R_spec = " << pars.R_spec << endl;
    cout << "\t - disper_angle = " << pars.disper_angle << endl;
    cout << "\t - offset = " << pars.offset << endl;
    cout << "\t - diameter [cm] = " << pars.diameter_in_cm << endl;
    cout << "\t - exptime [sec] = " << pars.exptime_in_sec << endl;
    cout << "\t - gain = " << pars.gain << endl;

    return 0;
}
//using namespace pybind11::literals;
int cpp_test_dict(const py::dict &d){
    for (auto item : d)
    {
        cout << "key: " << item.first << ", value=" << item.second << endl;
    }
    cout << py::int_(d["a"]) << endl;
    int a = py::int_(d["a"]);
    //cout << a << endl;
    double b = py::float_(d["b"]);
    string c = py::str(d["c"]);
    cout << a << b << c << endl;
    return 0;
}
/* Dispersion Relation
 * At the first call, the function would init the dispersion relation.
 * For a galaxy at real position (xcen,ycen), and with
 * dispersion angle theta, the wavelength lam gets dispersed
 * to the new position:
 *      x = xcen + (lam * dx/dlam + offset) * cos(theta),
 *      y = ycen + (lam * dx/dlam + offset) * sin(theta)
 * Input
 *      double lam: central wavelength in nm of the current slice
 *      vector<double> &shift: the returned resulting shift vector.
 * */

int cpp_dispersion_relation(double lam, vector<double> &shift,
                            const py::dict &config)
{
    static int INIT = 0;
    double dxdlam = double(py::float_(config["R_spec"])) / 500.0;
    vector<double> disp_vec{cos(double(py::float_(config["disp_ang"]))),
                            sin(double(py::float_(config["disp_ang"])))};
    double offset = py::float_(config["offset"]);
    shift[0] = (lam * dxdlam + offset) * disp_vec[0];
    shift[1] = (lam * dxdlam + offset) * disp_vec[1];
    return 0;
}
double obs2theory_arcsec(double center, int edge, double shift_in_pix,
                         double ref, const py::dict &config){
    double pix_scale = py::float_(config["pix_scale"]);
    return center+(edge*0.5-shift_in_pix)*pix_scale-ref;
}
/* Generate simulated grism image out of theory 3d model cube, but with CPP
 * implementation.
 *
 * THOUGHTS:
 *      As long as the dispersion relation is not changed and the theory_cube,
 *      lambdas, bandpass and image grid are the same, you can init the pixel
 *      response as a sparse matrix at the beginning, then during the MCMC
 *      sampling, you only need to distribute the updated theory_cube into the
 *      dispersed image according to this pixel response. This pixel response
 *      can include chromatic PSF.
 *
 * Inputs:
 *      theory_data: the 3D model cube of dimension
 *          (Nlam_theory, Ny_theory, Nx_theory). It contains the intensity
 *          distribution, in units of [photons/s/cm2]
 *      lambdas: the 2D array of dimension (Nlam_theory, 2), recording the
 *          wavelength edge of each model slice.
 *      bandpasses: the 2D array of dimension (Nlam_theory, 2), recording the
 *          bandpass at the wavelength edge of each model slice.
 * Outputs:
 *      dispersed_data: the 2D image stamp of dimension (Ny, Nx) after
 *          dispersing the 3D cube to a 2D (observed) image grid.
 * */

int cpp_stack(
  const py::array_t<double, py::array::c_style | py::array::forcecast> theory_data,
  const py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
  const py::array_t<double, py::array::c_style | py::array::forcecast> bandpasses,
  py::array_t<double, py::array::c_style | py::array::forcecast> dispersed_data,
  const py::dict &config)
{
  int i,j,k;
  double l,r,t,b,lb,rb,tb,bb;
  int li,ri,ti,bi;
  // unpack config params
  int model_Nx = py::int_(config["model_Nx"]);
  int model_Ny = py::int_(config["model_Ny"]);
  int model_Nlam = py::int_(config["model_Nlam"]);
  double model_scale = py::float_(config["model_scale"]);
  int Nx = py::int_(config["Nx"]);
  int Ny = py::int_(config["Ny"]);
  double pix_scale = py::float_(config["pix_scale"]);
  double diameter = py::float_(config["diameter"]);
  double exp_time = py::float_(config["exp_time"]);
  double gain = py::float_(config["gain"]);

  // sanity check
  py::buffer_info buf_td = theory_data.request();
  py::buffer_info buf_l = lambdas.request();
  py::buffer_info buf_bp = bandpasses.request();
  py::buffer_info buf_dd = dispersed_data.request();

  if(buf_td.ndim != 3)
    throw runtime_error("`theory_data` dimension must be 3!");
  if(buf_l.ndim != 2)
    throw runtime_error("`lambdas` dimension must be 2!");
  if(buf_bp.ndim != 2)
    throw runtime_error("`bandpasses` dimension must be 2!");
  if (buf_dd.ndim != 2)
    throw runtime_error("`dispersed_data` dimension must be 2!");
  if( (buf_td.shape[0] != buf_l.shape[0]) || \
      (buf_td.shape[0] != buf_bp.shape[0]) || \
      (buf_td.shape[0] != model_Nlam))
    throw runtime_error("`theory_data`, `lambdas` and `bandpasses` must have the same Nlam!");
  if((buf_td.shape[1] != model_Ny) || (buf_td.shape[2] != model_Nx))
    throw runtime_error("`theory_data` dimension wrong!");
  if(buf_dd.shape[0] != Ny || buf_dd.shape[1] != Nx)
    throw runtime_error("`dispersed_data` dimension wrong!");
  // get pointer to the buffer data memory
  auto *ptr_td = static_cast<double *>(buf_td.ptr);
  auto *ptr_l = static_cast<double *>(buf_l.ptr);
  auto *ptr_bp = static_cast<double *>(buf_bp.ptr);
  auto *ptr_dd = static_cast<double *>(buf_dd.ptr);

  // init coordinates
  // Note that those coordinates are static variables to save computation time.
  // theory model cube
  double Rx_theory = (int)(model_Nx/2) - 0.5 * ((model_Nx - 1) % 2);
  double Ry_theory = (int)(model_Ny/2) - 0.5 * ((model_Ny - 1) % 2);
  vector<double> origin_Xgrid(model_Nx, 0.0);
  vector<double> origin_Ygrid(model_Ny, 0.0);
  //cout << "Rx_theory = " << Rx_theory << endl;
  //cout << "Init X grid (theory cube): " << endl;
  for(i=0; i<model_Nx; i++){origin_Xgrid[i] = (i - Rx_theory) * model_scale;}
  //cout << endl;
  //cout << "Init Y grid (theory cube): " << endl;
  for(i=0; i<model_Ny; i++){origin_Ygrid[i] = (i - Ry_theory) * model_scale;}
  //cout << endl;
  double ob_x = origin_Xgrid[0] - 0.5*model_scale;
  double ob_y = origin_Ygrid[0] - 0.5*model_scale;
  //cout << "corner of the theory cube frame: "<< ob_x << ob_y << endl;
  // observed image
  double Rx = (int)(Nx/2) - 0.5 * ((Nx - 1) % 2);
  double Ry = (int)(Ny/2) - 0.5 * ((Ny - 1) % 2);
  vector<double> target_Xgrid(Nx, 0.0);
  vector<double> target_Ygrid(Ny, 0.0);
  //cout << "Rs_obs = " << Rx << endl;
  //cout << "Init X grid (observed image): " << endl;
  for(i=0; i<Nx; i++){target_Xgrid[i] = (i - Rx) * pix_scale;}
  //cout << endl << "Init Y grid (observed image): " << endl;
  for(i=0; i<Ny; i++){target_Ygrid[i] = (i - Ry) * pix_scale;}
  //cout << endl;
  // exptime calculation
  double flux_scale = PI*pow((diameter/2.0), 2)*exp_time/gain;

  // init dispersed_data
  for(size_t index = 0; index < buf_dd.size; index++){ptr_dd[index] = 0.0;}
  // looping through theory data cube
  for (i=0; i<model_Nlam; i++){
    vector<double> shift{0.0, 0.0}; // in units of pixel
    double blue_limit = ptr_l[2*i+0];
    double red_limit = ptr_l[2*i+1];
    double mean_wave = (blue_limit + red_limit)/2.;
    // double dlam = red_limit - blue_limit;
    // take the linear average of the bandpass.
    // Note that this only works when the lambda grid is fine enough.
    //double mean_bp = (bandpasses[2*i+0] + bandpasses[2*i+1])/2.0;
    double mean_bp = (ptr_bp[2*i+0] + ptr_bp[2*i+1])/2.0;
    // for each slice, disperse & interpolate
    cpp_dispersion_relation(mean_wave, shift, config);
    //cout << "slice " << i << " shift = (" << shift[0] << ", " << shift[1] << \
    //  ")" << "mean wavelength = " << mean_wave << endl;
    // loop through the dispersed image
    for(j=0; j<Ny; j++){
      for(k=0; k<Nx; k++){
        /* For each pixel in the dispersed image, find its original
         * pixels who contribute its flux. Then distribute the photons
         * from the theory cube to the observed image. If part of the
         * cell is involved, linear interpolation is applied.
         * */
        // For dispersed pixel (j,k), find its corners position in
        // arcsec, then map these corners to theory model cube, in units
        // of arcsec w.r.t. the lower-left corner of the theory model
        // cube pixel.

        l = obs2theory_arcsec(target_Xgrid[k], -1, shift[0], ob_x, config);
        r = obs2theory_arcsec(target_Xgrid[k], 1, shift[0], ob_x, config);
        b = obs2theory_arcsec(target_Ygrid[j], -1, shift[1], ob_y, config);
        t = obs2theory_arcsec(target_Ygrid[j], 1, shift[1], ob_y, config);
        lb = fmin(fmax(l/model_scale, 0), model_Nx);
        rb = fmin(fmax(r/model_scale, 0), model_Nx);
        bb = fmin(fmax(b/model_scale, 0), model_Ny);
        tb = fmin(fmax(t/model_scale, 0), model_Ny);
        li = floor(lb);
        ri = ceil(rb);
        bi = floor(bb);
        ti =  ceil(tb);
        // begin distribution
        if((li==ri) || (bi==ti)){continue;}//pixel outside the range
        else{
          int _nx = ri - li;
          int _ny = ti - bi;
          vector<double> x_weight(_nx, 1.0);
          vector<double> y_weight(_ny, 1.0);
          if(_nx > 1)
          {
            x_weight[0] = 1.0 + li - lb;
            x_weight[_nx-1] = 1.0 + rb - ri;
          }
          else{x_weight[0] = rb - lb;}

          if(_ny > 1)
          {
            y_weight[0] = 1.0 + bi - bb;
            y_weight[_ny-1] = 1.0 + tb - ti;
          }
          else{y_weight[0] = tb - bb;}
          // linear interpolation
          for(int p=0; p<_ny; p++)
          {
            for(int q=0; q<_nx; q++)
            {
              int _k = p + bi;
              int _l = q + li;
              size_t dd_id = dispersed_data.index_at(j,k);
              size_t td_id = theory_data.index_at(i,_k,_l);
              ptr_dd[dd_id] += ptr_td[td_id] * \
                x_weight[q]*y_weight[p]*mean_bp*flux_scale;
            }
          }
        }
        // end distribution
      }
    }
  }
  return 0;
}

void cpp_test_numpy_3d_array(
        py::array_t<double, py::array::c_style | py::array::forcecast> array)
{
  // play with NumPy array
  //cout << "First element of the array " << array[0] << endl;
  cout << "gets the count from the beginning to a given index: ";
  cout << array.index_at(1,1,1) << endl;
  // try access with buffer
  py::buffer_info buf = array.request();
  cout << "Buffer ndim = " << buf.ndim << endl;
  if(buf.ndim != 3)
    throw runtime_error("Number of dimension must be three!");
  cout << "Buffer size = " << buf.size << endl;
  cout << "Buffer shape = ";
  for (auto i: buf.shape)
    cout << i << ' ';
  cout << endl;
  cout << "Buffer format descriptor = " << buf.format << endl;
  cout << "Buffer strides = ";
  for (auto i: buf.strides)
    cout << i << " ";
  cout << endl;
  cout << "iterating using buffer ptr" << endl;
  auto *ptr = static_cast<double *>(buf.ptr);
  size_t index = 0;
  for(size_t i = 0; i < buf.shape[0]; i++) {
    for (size_t j = 0; j < buf.shape[1]; j++) {
      for (size_t k = 0; k < buf.shape[2]; k++) {
        //cout << ptr[index] << " ";
        ptr[index] = -1 * ptr[index];
        index++;
      }
    }
  }
  cout << endl;
}

/* PYBIND11 Python Wrapper
 * */
PYBIND11_MODULE(kltools_grism_module, m) {

  py::bind_vector<std::vector<double>>(m, "DBVec");

  m.doc() = "cpp grism module"; // optional module docstring

  m.def("set_pars", &cpp_set_pars, "A function that set Pars struct");

  m.def("print_Pars", &cpp_print_Pars, "A function that print Pars struct");

  m.def("stack", &cpp_stack,
        "A function that disperse and stack the theory model cube");

  m.def("test_numpy_3d_array", &cpp_test_numpy_3d_array,
        "A function that test the usage of NumPy array interface");
  m.def("test_dict", &cpp_test_dict, "");
  #ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
  #else
  m.attr("__version__") = "dev";
  #endif
}


