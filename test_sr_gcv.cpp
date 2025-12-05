#include <fdaPDE/models.h>
using namespace fdapde;

int main(){
    // geometry
    std::string mesh_path = "../fdaPDE-cpp/test/data/mesh/unit_square_21/";
    Triangulation<2, 2> D(mesh_path + "points.csv", mesh_path + "elements.csv", mesh_path + "boundary.csv", true, true);
    // data
    GeoFrame data(D);
    auto& l1 = data.insert_scalar_layer<POINT>("l1", MESH_NODES);
    l1.load_csv<double>("../fdaPDE-cpp/test/data/sr/04/response.csv");
    // physics
    FeSpace Vh(D, P1<1>);
    TrialFunction f(Vh);
    TestFunction  v(Vh);
    auto a = integral(D)(dot(grad(f), grad(v)));
    ZeroField<2> u;
    auto F = integral(D)(u * v);
    // modeling
    SRPDE m("y ~ f", data, fe_ls_elliptic(a, F));
    // calibration
    std::vector<double> lambda_grid(13);
    for (int i = 0; i < 13; ++i) { lambda_grid[i] = std::pow(10, -6.0 + 0.25 * i) / data[0].rows(); }
    GridSearch<1> optimizer;
    optimizer.optimize(m.gcv(100, 476813), lambda_grid);

    std::cout<<"ottimo"<<optimizer.optimum()<<"value:"<<optimizer.value();
    for (auto&  i : optimizer.values()){
    	std::cout<<i<<std::endl;
    }
 
    // EXPECT_TRUE(almost_equal<double>(optimizer.values(), "fdaPDE-cpp/test/data/sr/04/gcvs.mtx"));
}
