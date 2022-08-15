#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <cmath>
#include <cstdlib>
#include <string>
#include <complex>
#include <vector>
#include <time.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <omp.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)
#define _DEBUG_PRINTS_ 0

namespace py = pybind11;

#include "galsim/GSParams.h"
#include "galsim/SBInclinedExponential.h"
#include "galsim/Image.h"
#include "galsim/Random.h"
#define PI (3.14159265359)
using namespace std;

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
struct pixel_response {
    int image_x; // x index of dispersed image
    int image_y; // y index of dispersed image
    int cube_z; // z (wavelength) index of theory cube
    int cube_x; // x index of theory cube
    int cube_y; // y index of theory cube
    double weight; // weight of the theory cube
};

class disperse_helper {

public:
    disperse_helper(const py::dict &config,
                    const py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
                    const py::array_t<double, py::array::c_style | py::array::forcecast> bandpasses);
    void set_disperse_helper(const py::dict &config,
                             const py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
                             const py::array_t<double, py::array::c_style | py::array::forcecast> bandpasses);
    void get_dispersed_image(
        const py::array_t<double, py::array::c_style | py::array::forcecast> theory_data,
        py::array_t<double, py::array::c_style | py::array::forcecast> dispersed_data);

private:
    // configuration parameters
    int model_Nx, model_Ny, model_Nlam; // theory model cube dimension
    double model_scale; // theory model cube pixel scale
    int Nx, Ny; // observed image dimension
    double pix_scale; // observed image pixel scale
    double R_spec; // grism spectral resolution at 1 micron
    double disp_ang; // dispersion angle, radian
    double offset; // offset in units of observed pixels
    double diameter; // aperture diameter in cm
    double exp_time; // exposure time in seconds
    double gain; // detector gain
    // disperse relation table
    vector<pixel_response> pixel_response_table;
    int set_pixel_response(
            const py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
            const py::array_t<double, py::array::c_style | py::array::forcecast> bandpasses);
    void get_dispersion(double lam, vector<double> &shift);
    double img2cube_arcsec(double center, int edge, double shift_in_pix,
                           double ref){return center+(edge*0.5-shift_in_pix)*pix_scale-ref;}
};

disperse_helper::disperse_helper(const py::dict &config,
     const py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
     const py::array_t<double, py::array::c_style | py::array::forcecast> bandpasses)
 {
    model_Nx = py::int_(config["model_Nx"]);
    model_Ny = py::int_(config["model_Ny"]);
    model_Nlam = py::int_(config["model_Nlam"]);
    model_scale = py::float_(config["model_scale"]);
    Nx = py::int_(config["Nx"]);
    Ny = py::int_(config["Ny"]);
    pix_scale = py::float_(config["pix_scale"]);
    R_spec = py::float_(config["R_spec"]);
    disp_ang = py::float_(config["disp_ang"]);
    offset = py::float_(config["offset"]);
    diameter = py::float_(config["diameter"]);
    exp_time = py::float_(config["exp_time"]);
    gain = py::float_(config["gain"]);
    int status = set_pixel_response(lambdas, bandpasses);
}

void disperse_helper::set_disperse_helper(const py::dict &config,
      const py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
      const py::array_t<double, py::array::c_style | py::array::forcecast> bandpasses)
{
    model_Nx = py::int_(config["model_Nx"]);
    model_Ny = py::int_(config["model_Ny"]);
    model_Nlam = py::int_(config["model_Nlam"]);
    model_scale = py::float_(config["model_scale"]);
    Nx = py::int_(config["Nx"]);
    Ny = py::int_(config["Ny"]);
    pix_scale = py::float_(config["pix_scale"]);
    R_spec = py::float_(config["R_spec"]);
    disp_ang = py::float_(config["disp_ang"]);
    offset = py::float_(config["offset"]);
    diameter = py::float_(config["diameter"]);
    exp_time = py::float_(config["exp_time"]);
    gain = py::float_(config["gain"]);
    // update the pixel response
    int status = set_pixel_response(lambdas, bandpasses);
}

void disperse_helper::get_dispersion(double lam, vector<double> &shift) {
    shift[0] = (lam * (R_spec/500.0) + offset) * cos(disp_ang);
    shift[1] = (lam * (R_spec/500.0) + offset) * sin(disp_ang);
}

int disperse_helper::set_pixel_response(
        const py::array_t<double, py::array::c_style | py::array::forcecast> lambdas,
        const py::array_t<double, py::array::c_style | py::array::forcecast> bandpasses
        )
{
    int i,j,k;
    double l,r,t,b,lb,rb,tb,bb;
    int li,ri,ti,bi;
    // sanity check on lambdas array
    py::buffer_info buf_l = lambdas.request();
    py::buffer_info buf_bp = bandpasses.request();
    if(buf_l.ndim != 2)
        throw runtime_error("`lambdas` dimension must be 2!");
    if(buf_bp.ndim != 2)
        throw runtime_error("`bandpasses` dimension must be 2!");
    if((buf_l.shape[0] != model_Nlam) || (buf_bp.shape[0] != model_Nlam))
        throw runtime_error("`lambdas` has wrong Nlam!");
    auto *ptr_l = static_cast<double *>(buf_l.ptr);
    auto *ptr_bp = static_cast<double *>(buf_bp.ptr);

    /*************** start pixel response calculation ******************/
    // init coordinates
    // theory model cube
    double Rx_theory = (int)(model_Nx/2) - 0.5 * ((model_Nx - 1) % 2);
    double Ry_theory = (int)(model_Ny/2) - 0.5 * ((model_Ny - 1) % 2);
    vector<double> origin_Xgrid(model_Nx, 0.0);
    vector<double> origin_Ygrid(model_Ny, 0.0);
    for(i=0; i<model_Nx; i++){origin_Xgrid[i] = (i - Rx_theory) * model_scale;}
    for(i=0; i<model_Ny; i++){origin_Ygrid[i] = (i - Ry_theory) * model_scale;}
    double ob_x = origin_Xgrid[0] - 0.5*model_scale;
    double ob_y = origin_Ygrid[0] - 0.5*model_scale;
    // observed image
    double Rx = (int)(Nx/2) - 0.5 * ((Nx - 1) % 2);
    double Ry = (int)(Ny/2) - 0.5 * ((Ny - 1) % 2);
    vector<double> target_Xgrid(Nx, 0.0);
    vector<double> target_Ygrid(Ny, 0.0);
    for(i=0; i<Nx; i++){target_Xgrid[i] = (i - Rx) * pix_scale;}
    for(i=0; i<Ny; i++){target_Ygrid[i] = (i - Ry) * pix_scale;}
    if(_DEBUG_PRINTS_){
        cout << "Rx_theory = " << Rx_theory << endl;
        cout << "Init X grid (theory cube): " << endl;
        for (auto item : origin_Xgrid)
            cout << item << " " << endl;
        cout << endl << "Ry_theory = " << Ry_theory << endl;
        cout << "Init Y grid (theory cube): " << endl;
        for (auto item : origin_Ygrid)
            cout << item << " " << endl;
        cout << endl;
        cout << "corner of the theory cube frame: "<< ob_x << ob_y << endl;
        cout << "Rs_obs = " << Rx << endl;
        cout << "Init X grid (observed image): " << endl;
        for (auto item : target_Xgrid)
            cout << item << " " << endl;
        cout << endl << "Ry_obs = " << Ry << endl;
        cout << endl << "Init Y grid (observed image): " << endl;
        for (auto item : target_Ygrid)
            cout << item << " " << endl;
        cout << endl;
    }
    // exptime calculation
    double flux_scale = PI*pow((diameter/2.0), 2)*exp_time/gain;
    // looping through theory data cube
    cout << "Setting pixel response table..." << endl;
    cout << "Theory model cube dimension: (scale = " << model_scale << " )";
    cout << "("<< model_Nlam << ", " << model_Ny << ", " << model_Nx << ")" << endl;
    cout << "Dispersed image dimension: (scale = " << pix_scale <<  ")";
    cout << "(" << Ny << ", " << Nx << ")" << endl;
    if(pixel_response_table.size() > 0){pixel_response_table.clear();}
    for (i=0; i<model_Nlam; i++){
        vector<double> shift{0.0, 0.0}; // in units of pixel
        double blue_limit = ptr_l[2*i+0];
        double red_limit = ptr_l[2*i+1];
        double mean_wave = (blue_limit + red_limit)/2.;
        // take the linear average of the bandpass.
        // Note that this only works when the lambda grid is fine enough.
        double mean_bp = (ptr_bp[2*i+0] + ptr_bp[2*i+1])/2.0;
        // for each slice, disperse & interpolate
        get_dispersion(mean_wave, shift);
        if(_DEBUG_PRINTS_)
        {
            cout << "slice " << i << " shift = (" << shift[0] << \
            ", " << shift[1] << ")" << "mean wavelength = " \
            << mean_wave << endl;
        }

        // loop through the dispersed image
        for(j=0; j<Ny; j++){
            for(k=0; k<Nx; k++){
                // For each pixel in the dispersed image, find its original
                // pixels who contribute its flux. Then distribute the photons
                // from the theory cube to the observed image. If part of the
                // cell is involved, linear interpolation is applied.
                // For dispersed pixel (j,k), find its corners position in
                // arcsec, then map these corners to theory model cube, in units
                // of arcsec w.r.t. the lower-left corner of the theory model
                // cube pixel.

                l = img2cube_arcsec(target_Xgrid[k], -1, shift[0], ob_x);
                r = img2cube_arcsec(target_Xgrid[k], 1, shift[0], ob_x);
                b = img2cube_arcsec(target_Ygrid[j], -1, shift[1], ob_y);
                t = img2cube_arcsec(target_Ygrid[j], 1, shift[1], ob_y);
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
                            // record the response here
                            // dispersed image index: y=j, x=k
                            // theory cube index: lam=i, y=_k, x=_l
                            // weight: x_weight[q]*y_weight[p]*mean_bp*flux_scale
                            pixel_response _res;
                            _res.image_x = k;
                            _res.image_y = j;
                            _res.cube_x = _l;
                            _res.cube_y = _k;
                            _res.cube_z = i;
                            _res.weight = x_weight[q]*y_weight[p]*mean_bp*flux_scale;
                            pixel_response_table.push_back(_res);
                        }
                    }
                }
                // end distribution
            }// End x-loop, obs image
        }// End y-loop, obs image
    }// End lambda-loop, theory cube
    cout << "Pixel res. table size = " << pixel_response_table.size() << endl;
    return 0;
}

void disperse_helper::get_dispersed_image(
        const py::array_t<double, py::array::c_style | py::array::forcecast> theory_data,
        py::array_t<double, py::array::c_style | py::array::forcecast> dispersed_data)
{
    // sanity check
    py::buffer_info buf_td = theory_data.request();
    py::buffer_info buf_dd = dispersed_data.request();

    if(buf_td.ndim != 3)
        throw runtime_error("`theory_data` dimension must be 3!");
    if(buf_dd.ndim != 2)
        throw runtime_error("`dispersed_data` dimension must be 2!");
    if(buf_td.shape[0] != model_Nlam)
        throw runtime_error("`theory_data`, must have the same Nlam!");
    if((buf_td.shape[1] != model_Ny) || (buf_td.shape[2] != model_Nx))
        throw runtime_error("`theory_data` dimension wrong!");
    if(buf_dd.shape[0] != Ny || buf_dd.shape[1] != Nx)
        throw runtime_error("`dispersed_data` dimension wrong!");
    // get pointer to the buffer data memory
    auto *ptr_td = static_cast<double *>(buf_td.ptr);
    auto *ptr_dd = static_cast<double *>(buf_dd.ptr);

    // init dispersed_data
    for(size_t index = 0; index < buf_dd.size; index++){ptr_dd[index] = 0.0;}
    // begin distribution
    unsigned int thread_qty = max(atoi(getenv("OMP_NUM_THREADS")), 1);
    omp_set_num_threads(thread_qty);

#pragma omp parallel shared(dispersed_data, theory_data, ptr_dd, \
    ptr_td, pixel_response_table)
{
    vector<double> local_copy(Ny*Nx, 0.0);
    #pragma omp for
    for (int k = 0; k < pixel_response_table.size(); k++) {
        const auto &item = pixel_response_table[k];
        // record the response here
        // dispersed image index: y=j, x=k
        // theory cube index: lam=i, y=_k, x=_l
        // weight: x_weight[q]*y_weight[p]*mean_bp*flux_scale
        size_t local_copy_id = item.image_y * Nx + item.image_x;
        size_t td_id = theory_data.index_at(item.cube_z, item.cube_y, item.cube_x);

        local_copy[local_copy_id] += ptr_td[td_id] * item.weight;
    }
    for (int j=0; j<Ny; j++){
        for(int i=0; i<Nx; i++){
            size_t dd_id = dispersed_data.index_at(j, i);
            #pragma omp critical
            {
                ptr_dd[dd_id] += local_copy[j * Nx + i];
            }
        }
    }
}
}

/* PYBIND11 Python Wrapper
 * */
PYBIND11_MODULE(kltools_grism_module_2, m) {

  m.doc() = "cpp grism module"; // optional module docstring

  py::class_<disperse_helper>(m, "DisperseHelper")
    .def(py::init<const py::dict &, const py::array_t<double, py::array::c_style | py::array::forcecast>, const py::array_t<double, py::array::c_style | py::array::forcecast> >())
    .def("setDisperseHelper", &disperse_helper::set_disperse_helper)
    .def("getDispersedImage", &disperse_helper::get_dispersed_image);

  #ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
  #else
  m.attr("__version__") = "dev";
  #endif
}


