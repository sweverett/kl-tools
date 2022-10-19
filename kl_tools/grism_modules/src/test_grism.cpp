// exec file testing GalSim
// Created by jiachuan on 7/7/22.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <complex>
#include <vector>
#include <time.h>
#include "galsim/GSParams.h"
#include "galsim/SBInclinedExponential.h"
#include "galsim/Image.h"
#include "galsim/Random.h"

#define idx( i, Ni, j, Nj ) ((int)( (i) * (Nj) + (j) ))
#define PI (3.14159265359)
using namespace std;

int test_galsim()
{
    /**
     * This part of the code is to reproduce the following Python code

     import galsim as gs
     from galsim import Angle, radians

     inc = Angle(np.pi/6., unit=radians)
     rs = 0.4
     hs = 0.04
     flux = 1.0
     pix_scale = 0.065
     Nx = 38
     Ny = 38

     gal = gs.InclinedExponential(inc, scale_radius = rs, scale_height = hs,
                                  flux = flux)

     img = gal.drawImage(nx=Nx, ny=Ny, scale=pix_scale, method='fft',
                          bandpass=None)
     */
    // Example galaxy parameters
    double inc = PI/6.;
    double rs = 0.4;
    double hs = 0.04;
    double flux = 1.0;
    double pix_scale = 0.065;
    double Nx = 38;
    double Ny = 38;
    cout << "Model Galaxy Parameters" << endl;
    cout << "- Inclination = " << inc << endl;
    cout << "- Scale radius = " << rs << endl;
    cout << "- Scale height = " << hs << endl;
    cout << "- Flux = " << flux << endl;

    // Create object
    cout << "Creating Object" << endl;
    galsim::GSParams gsparams;
    // Usage: SBInclinedExponential(
    // double inclination, double scale_radius, double scale_height,
    // double flux, const GSParams& gsparams);
    galsim::SBInclinedExponential gal(inc, rs, hs, flux, gsparams);

    // Create image
    cout << "Creating Image" << endl;
    // Usage: ImageAlloc(int ncol, int nrow, T init_values);
    galsim::ImageAlloc<double> img(Ny, Nx, 0.0);

    // Create RNG
    cout << "Creating RNG" << endl;
    galsim::BaseDeviate rng1(5678);

    // Create sensor ?
    cout << "Creating Sensor" << endl;

    // Draw an Image
    // Note that InclinedExponential does not have real space draw method
    // Usage: template <typename T>
    //        void draw(ImageView<T> image, double dx, double* jac,
    //                  double xoff, double yoff, double flux_ratio) const;
    // jac is "jacobian"
    gal.draw(img.view(), pix_scale, NULL, 0.0, 0.0, 1.0);

    // Save image data in raw binary
    cout << "Saving result" << endl;
    double* imageData = img.getData();
    int imageXMin = img.getXMin();
    int imageYMin = img.getYMin();
    int imageXMax = img.getXMax();
    int imageYMax = img.getYMax();
    int imageStep = img.getStep();
    int imageStride = img.getStride();

    ofstream out("../../output.bin", ios::binary | ios::out);
    if (out.fail()) {
        cerr << "Unable to open ../../output.bin for writing" << endl;
        return 1;
    }
    for (int y = 0; y < (imageYMax - imageYMin); y++) {
        for (int x = 0; x < (imageXMax - imageXMin); x++) {
            int idx = (x * imageStep) + (y * imageStride);
            out.write((const char *)&imageData[idx], sizeof(double));
        }
    }
    out.close();

    std::cout << "Done" << std::endl;

    return 0;
}
void disperse(const vector<double>& theory_slice, const vector<double>& shift,
              float scale, int Nx_theory, int Ny_theory,
              float pix_scale, int Nx, int Ny,
              vector<double> &target_img,
              const vector<float>& target_Xmesh, const vector<float>& target_Ymesh,
              const vector<float>& origin_Xmesh, const vector<float>& origin_Ymesh)
{
    vector<double> shift_arcsec = {
            shift[0] * scale,
            shift[0] * scale,
            shift[1] * scale,
            shift[1] * scale,
    };
    int i,j,k,l;
    double _target_arcsec[4]; // x left, x right, y bottom, y top
    double _origin_arcsec[4]; // x left, x right, y bottom, y top
    double _origin_pixel[4];  // x left, x right, y bottom, y top
    double _pixel_bound[4];   // x left, x right, y bottom, y top
    double _origin_corner[4] = {
            origin_Xmesh[0] - 0.5 * scale,
            origin_Xmesh[0] - 0.5 * scale,
            origin_Ymesh[0] - 0.5 * scale,
            origin_Ymesh[0] - 0.5 * scale,
    };
    double *x_weight, *y_weight;
    int _pixel_impact[4];     // x left, x right, y bottom, y top

    for(i=0; i<Ny; i++)
    {
        for(j=0; j<Nx; j++)
        {
            // target pixel in observed detector, [arcsec]
            // origin: center
            _target_arcsec[0] = target_Xmesh[j] - 0.5*pix_scale;
            _target_arcsec[1] = target_Xmesh[j] + 0.5*pix_scale;
            _target_arcsec[2] = target_Ymesh[i] - 0.5*pix_scale;
            _target_arcsec[3] = target_Ymesh[i] + 0.5*pix_scale;
            // target pixel -> theory model "origin pixel", [arcsec]
            // origin: lower-left
            for(k=0; k<4; k++)
            {
                _origin_arcsec[k] = _target_arcsec[k] - shift_arcsec[k] - _origin_corner[k];
                // origin pixel [arcsec] -> [pixel]
                _origin_pixel[k] = _origin_arcsec[k] / scale;
            }
            // bounded by the theory model cube boundary
            _pixel_bound[0] = fmin(fmax(_origin_pixel[0], 0), Nx_theory);
            _pixel_bound[1] = fmin(fmax(_origin_pixel[1], 0), Nx_theory);
            _pixel_bound[2] = fmin(fmax(_origin_pixel[2], 0), Ny_theory);
            _pixel_bound[3] = fmin(fmax(_origin_pixel[3], 0), Ny_theory);
            // origin: lower-left, starting from 0, [start, end)
            _pixel_impact[0] = floor(_pixel_bound[0]);
            _pixel_impact[1] =  ceil(_pixel_bound[1]);
            _pixel_impact[2] = floor(_pixel_bound[2]);
            _pixel_impact[3] =  ceil(_pixel_bound[3]);

            if((_pixel_impact[0]==_pixel_impact[1]) || (_pixel_impact[2]==_pixel_impact[3]))
            {
                continue;
            }
            else
            {
                int _nx = _pixel_impact[1]-_pixel_impact[0];
                int _ny = _pixel_impact[3]-_pixel_impact[2];
                x_weight = (double *) calloc(_nx, sizeof(double));
                y_weight = (double *) calloc(_ny, sizeof(double));
                if(_nx > 1)
                {
                    x_weight[0] = 1.0 + _pixel_impact[0] - _pixel_bound[0];
                    x_weight[_nx-1] = 1.0 + _pixel_bound[1] - _pixel_impact[1];
                    for(k=1; k<_nx-1; k++){x_weight[k] = 1.0;}
                }
                else
                {
                    x_weight[0] = _pixel_bound[1] - _pixel_bound[0];
                }
                if(_ny > 1)
                {
                    y_weight[0] = 1.0 + _pixel_impact[2] - _pixel_bound[2];
                    y_weight[_ny-1] = 1.0 + _pixel_bound[3] - _pixel_impact[3];
                    for(k=1; k<_ny-1; k++){y_weight[k] = 1.0;}
                }
                else
                {
                    y_weight[0] = _pixel_bound[3] - _pixel_bound[2];
                }
                for(k=0; k<_ny; k++)
                {
                    for(l=0; l<_nx; l++)
                    {
                        int _k = k + _pixel_impact[2];
                        int _l = l + _pixel_impact[0];
                        target_img[idx(i,Ny,j,Nx)] += theory_slice[idx(_k,Ny_theory,_l,Nx_theory)] * x_weight[l] * y_weight[k];
                    }
                }
                free(x_weight);
                free(y_weight);
            }

        }
    }
}

int main()
{
    /*
	int i,j,k,l;
	// theory model cube
	float scale = 0.05;
	int Nx_theory = 60, Ny_theory=60;
	float Rx_theory = (int)(Nx_theory/2) - 0.5 * ((Nx_theory - 1) % 2);
	float Ry_theory = (int)(Ny_theory/2) - 0.5 * ((Ny_theory - 1) % 2);
	float *origin_Xgrid = (float *)calloc(Nx_theory, sizeof(float));
	float *origin_Ygrid = (float *)calloc(Ny_theory, sizeof(float));
	for(i=0; i<Nx_theory; i++){
		origin_Xgrid[i] = (Rx_theory + i) * scale;
	}
	for(i=0; i<Ny_theory; i++){
		origin_Ygrid[i] = (Ry_theory + i) * scale;
	}

	// detector pixel
	float pix_scale = 0.065;
	int Nx = 38, Ny=38;
	float Rx = (int)(Nx/2) - 0.5 * ((Nx - 1) % 2);
	float Ry = (int)(Ny/2) - 0.5 * ((Ny - 1) % 2);
	float *target_Xgrid = (float *)calloc(Nx, sizeof(float));
	float *target_Ygrid = (float *)calloc(Ny, sizeof(float));
	for(i=0; i<Nx; i++){
		target_Xgrid[i] = (Rx + i) * pix_scale;
	}
	for(i=0; i<Ny; i++){
		target_Ygrid[i] = (Ry + i) * pix_scale;
	}
	double *target_img = (double *)calloc(Nx * Ny, sizeof(double));

	// lambda grid
	float lambda_blue = 1236.6, lambda_red = 1325.1;
	float lambda_res = 0.05; // nm
	int Nlam = (int)((lambda_red - lambda_blue)/lambda_res);
	double *lambdas = (double *)calloc(Nlam, sizeof(double));

	// grism related
	double disp_ang = 0.0; // rad, 0, 1.57
	double dxdlam = 4.65; // nm / pixel
	double offset = -275.48161224045805; // pixel
	double *shift = (double *)calloc(2*Nlam, sizeof(double));

	for(i=0; i<Nlam; i++)
	{
		lambdas[i] = lambda_blue + (i+0.5)*lambda_res;
		shift[i*2]   = (lambdas[i] * dxdlam + offset) * cos(disp_ang);
		shift[i*2+1] = (lambdas[i] * dxdlam + offset) * sin(disp_ang);
	}

	// theory cube
	double *theory_slice = (double *)calloc(Nlam * Ny_theory * Nx_theory, sizeof(double));

	printf("Theory model cube size (%d, %d, %d)\n", Nx_theory, Ny_theory, Nlam);
	// run dispersion
	clock_t start = clock();
	for(i=0; i<Nlam; i++)
	{
		double *_slice = theory_slice + i * Nx_theory * Ny_theory;
		double *_shift = shift + i * 2;
		disperse(_slice, _shift,
			scale, Nx_theory, Ny_theory,
			pix_scale, Nx, Ny, target_img,
			target_Xgrid, target_Ygrid,
			origin_Xgrid, origin_Ygrid);
	}
	clock_t stop = clock();
	double time_spent = (double)(stop - start) / CLOCKS_PER_SEC;
	printf("Done (%f ms)\n", time_spent*1000.);
    */
    test_galsim();
}