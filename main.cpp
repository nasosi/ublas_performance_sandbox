#include <iostream>
#include <chrono>
#include <Eigen/Dense>
#include <boost/numeric/ublas/matrix.hpp>
#include <omp.h>
#include <sstream>
#include <iomanip>
#include <limits>

#define COMPUTE_STRIDE_SIZE 32
#define COMPUTE_STRIDE_SIZE2 2

typedef double value_type;
typedef boost::numeric::ublas::matrix<value_type> ublas_matrix_type;
typedef Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> eigen_matrix_type;

std::chrono::high_resolution_clock::time_point now() {
    return std::chrono::high_resolution_clock::now();
}

double duration_since( const std::chrono::high_resolution_clock::time_point &since) {
    return std::chrono::duration_cast<std::chrono::microseconds>(now() - since).count();
}


class timer {
public:
    timer():paused(false) { restart(); }

    void restart() {
        start_time_point_ = std::chrono::high_resolution_clock::now();
    }

    void cont() {
        if (paused) {
            paused = false;
           start_time_point_ += std::chrono::high_resolution_clock::now() - paused_time_point_;
        }

    }

    std::chrono::high_resolution_clock::duration  elapsed() {
        auto now = std::chrono::high_resolution_clock::now();
        if (paused) return paused_time_point_ - start_time_point_;
        return now - start_time_point_;
    }

    long elapsed_in_microseconds() {
        return std::chrono::duration_cast<std::chrono::microseconds>(elapsed()).count();
    }

    void pause() {
        paused_time_point_ = std::chrono::high_resolution_clock::now();
        paused = true;
    }

private:
    bool paused;
    std::chrono::high_resolution_clock::time_point paused_time_point_;
    std::chrono::high_resolution_clock::time_point start_time_point_;
};

using Eigen::MatrixXd;



inline void axpy_local(value_type a, const value_type *  __restrict__ xx, value_type *  __restrict__ yy )
{
    const value_type *x = static_cast<const value_type*>(__builtin_assume_aligned(xx, 16));
    value_type *y = static_cast<value_type*>(__builtin_assume_aligned(yy, 16));

    for (std::size_t i = 0; i < COMPUTE_STRIDE_SIZE; i++)
        y[i]+= a*x[i];
}

inline void axpy_local22(value_type *  __restrict__ yy, const value_type *  __restrict__ xx, value_type a)
{
    value_type *x = static_cast<value_type*>(__builtin_assume_aligned(xx, 16));
    value_type *y = static_cast<value_type*>(__builtin_assume_aligned(yy, 16));

    for (int i = 0; i < COMPUTE_STRIDE_SIZE2; i++)
        y[i]+= a*x[i];
}

inline void axpy_local2(value_type y[COMPUTE_STRIDE_SIZE], const value_type x[COMPUTE_STRIDE_SIZE], value_type a)
{

    for (std::size_t i = 0; i < COMPUTE_STRIDE_SIZE; i++)
        y[i]+= a*x[i];
}


void axpy(value_type *  yy, const value_type *  xx, value_type a, std::size_t size)
{
    const value_type *x = static_cast<const value_type*>(__builtin_assume_aligned(xx, 16));
    value_type *y = static_cast<value_type*>(__builtin_assume_aligned(yy, 16));

    for (std::size_t i = 0; i < size; i+=COMPUTE_STRIDE_SIZE)
        axpy_local(a, x+i,  y+i);
}


template <class T>
void axpy2(const T &a, const boost::numeric::ublas::matrix<T> &x, boost::numeric::ublas::matrix<T> &y) {
    asm("# Begin axpy2");
    typedef T* pointer;
    typedef typename boost::numeric::ublas::matrix<T>::size_type size_type;

    T ya[COMPUTE_STRIDE_SIZE2], xa[COMPUTE_STRIDE_SIZE2];

    T *yy = &y(0,0);
    std::size_t size =  y.size1()*y.size2();
    const T *end =  &x(0,0) + size;

    for( volatile const T *xx=&x(0,0) ; xx< end; xx+=COMPUTE_STRIDE_SIZE2, yy+=COMPUTE_STRIDE_SIZE2) {
        for (int j=0; j<COMPUTE_STRIDE_SIZE2; j++) { ya[j] = yy[j]; xa[j] = xx[j]; }
        asm("# Begin local");
        axpy_local22(&xa[0], &ya[0], a);
        asm("# End local");
        for (int j=0; j<COMPUTE_STRIDE_SIZE2; j++) { yy[j] = ya[j]; }

    }
    asm("# End axpy2");

}


template <class T>
void axpy(const T &a, const boost::numeric::ublas::matrix<T> &x, boost::numeric::ublas::matrix<T> &y) {
    axpy(&y(0,0), &x(0,0), a, y.size1()*y.size2());
}

template <class T>
void paxpy(const T &a, const boost::numeric::ublas::matrix<T> &x, boost::numeric::ublas::matrix<T> &y) {

    typedef T* pointer;
    typedef typename boost::numeric::ublas::matrix<T>::size_type size_type;

    pointer yp = static_cast<pointer>(__builtin_assume_aligned(&y(0,0), 16));
    const pointer xp = static_cast<const pointer>(__builtin_assume_aligned(&x(0,0), 16));

    std::size_t s =  y.size1()*y.size2();
#pragma omp parallel for
    for (std::size_t i = 0; i < s; i+=COMPUTE_STRIDE_SIZE)
        axpy_local(a, xp+i, yp+i);
}

// The following are not efficiently optimized
template <class T, class M>
void axpy_local2(const T &a, const M &x, M &y, std::size_t i0, std::size_t j0) {
    for (std::size_t j = j0; j < j0+COMPUTE_STRIDE_SIZE; j++) {
        //std::cout << i0 << ", " << j << std::endl;
        y(i0, j) += a* x(i0, j);
    }
}
template <class T>
void paxpy2(const T &a, const boost::numeric::ublas::matrix<T> &x, boost::numeric::ublas::matrix<T> &y) {
#pragma omp parallel for
    for (std::size_t i = 0; i < y.size1(); i++)
        for (std::size_t j = 0; j < y.size2(); j+=COMPUTE_STRIDE_SIZE) {
           // std::cout << i << ", " << j << std::endl;
            axpy_local2(a, x, y, i, j);
        }
}

template <class Functor>
class test {
public:
    test(const Functor &f, double d): functor(f), duration(d) { }

    double execute() {
        functor.prepare();
        std::size_t count(0);
        timer t;

        t.restart();
        //t.pause();
        while( std::abs(t.elapsed_in_microseconds())< duration) {
            //std::cout << t.elapsed_in_microseconds() << ", " << count << std::endl;
            //t.cont();
            functor.doit();
            //t.pause();
            count++;
        }

        double elapsed = t.elapsed_in_microseconds();
        return (double)elapsed / count;
    }
private:
    Functor functor;
    std::size_t duration;
};

template <class TestType>
double runTest(const TestType &t, std::size_t min_duration = 100000) {
    test<TestType> tst(t, min_duration);
    return tst.execute();
}

const int SEED = 4356;

template <class T>
T myrand(T min=0.0, T max=1.0) {
    if (min>max) std::swap(min,max);
    return ((T)rand())/RAND_MAX*(max-min)+min;
}

template <class Matrix_Type>
void fill_random_eigen(Matrix_Type &M, value_type min=0.0, value_type max=1.0, int seed=-1) {

    if (seed<0) {
        srand ( time(NULL) );
    } else {
        srand( seed);
    }
    typedef std::size_t size_type;
    typedef value_type value_type;

    for (size_type i=0; i!=M.cols(); i++)
        for (size_type j=0; j!=M.rows(); j++)
            M(i,j) = myrand<value_type>(min, max);
}


template <class Matrix_Type>
void fill_random(Matrix_Type &M, value_type min=0.0, value_type max=1.0, int seed=-1) {

    if (seed<0) {
        srand ( time(NULL) );
    } else {
        srand( seed);
    }
    typedef typename Matrix_Type::size_type size_type;
    typedef typename Matrix_Type::value_type value_type;

    for (size_type i=0; i!=M.template size1(); i++)
        for (size_type j=0; j!=M.template size2(); j++)
            M(i,j) = myrand<value_type>(min, max);
}


class ublas_base  {
public:
    ublas_base(std::size_t s): size(s) { }

    std::string name() const { return "ublas base"; }

        int doit() {
            this->A += 2.54343*this->B;
        return -1;
    }


    ublas_base(const ublas_base &other): size(other.size) { }

    void prepare(int SEED = 4656) {
        A.resize(size, size);
        B.resize(size, size);

        fill_random(A, 0.0, 1.0, SEED);
        fill_random(B, 0.0, 1.0, SEED+1);

    }

    value_type get() {
        return A(0.75*size,0.25*size);
    }

    value_type sample_output(int SEED = 4656) {
        prepare(SEED);
        doit();
        return A(0.75*size,0.25*size);
    }

    const std::size_t &get_size() const { return size; }
protected:
    std::size_t size;
    ublas_matrix_type A, B;

};

class ublas_noalias: public ublas_base  {
public:
    ublas_noalias(std::size_t s): ublas_base(s) { }
    std::string name() const { return "ublas noalias"; }
        int doit() {
            boost::numeric::ublas::noalias(this->A) += 2.54343*this->B;
        return -1;
    }
};

class ublas_axpy: public ublas_base  {
public:
    ublas_axpy(std::size_t s): ublas_base(s) { }
    std::string name() const { return "ublas axpy"; }
        int doit() {
            axpy((value_type)2.54343, this->A, this->B);
        return -1;
    }
};

class ublas_axpy2: public ublas_base  {
public:
    ublas_axpy2(std::size_t s): ublas_base(s) { }
    std::string name() const { return "ublas axpy2"; }
        int doit() {
            axpy2((value_type)2.54343, this->A, this->B);
        return -1;
    }
};

class ublas_paxpy: public ublas_base  {
public:
    ublas_paxpy(std::size_t s, std::size_t nt): ublas_base(s), num_threads_(nt) { omp_set_num_threads(nt);}

    ublas_paxpy(const ublas_paxpy &other): ublas_base(other), num_threads_(other.num_threads_) { }

    std::string name() const {
        std::stringstream ss;
        ss << "ublas paxpy threads=" << num_threads_;
        return ss.str(); }
        int doit() {
            if (omp_get_max_threads()==1 || this->A.size1()*this->A.size2()<4096)
                axpy((value_type)2.54343, this->A, this->B);
            else
                paxpy((value_type)2.54343, this->A, this->B);
        return -1;
    }
private:
    std::size_t num_threads_;
};

//! EIGEN3 axpy ----------------------------------------------------------------
class eigen3  {
public:
    eigen3(std::size_t s): size(s) { }

    std::string name() const { return "Eigen 3"; }

        int doit() {
            this->A += 2.54343*this->B;
        return -1;
    }


    eigen3(const eigen3 &other): size(other.size) { }

    void prepare(int SEED = 4656) {
        A.resize(size, size);
        B.resize(size, size);

        fill_random_eigen(A, 0.0, 1.0, SEED);
        fill_random_eigen(B, 0.0, 1.0, SEED+1);

    }

    value_type get() {
        return A(0.75*size,0.25*size);
    }

    value_type sample_output(int SEED = 4656) {
        prepare(SEED);
        doit();
        return A(0.75*size,0.25*size);
    }

    const std::size_t &get_size() const { return size; }
protected:
    std::size_t size;
    eigen_matrix_type A, B;

};

double mflops(std::size_t size, double duration) {
        return (size*size)/(duration);
}

void print_results(std::size_t size, std::vector<double> elapsed) {
    std::ios_base::fmtflags original_flags = std::cout.flags();
    std::cout <<  std::setw(6) << (double)size*size*(sizeof(value_type))/1024/1024 << ',' << std::setfill(' ')<<  std::setprecision(4) << std::setw(11) << std::right <<4*mflops(size, elapsed[0]);
    for (std::size_t i=1; i!=elapsed.size(); i++)
        std::cout << ',' << std::setfill(' ')<<  std::setprecision(4) << std::setw(11) << std::right << 4*mflops(size, elapsed[i]);
    std::cout << std::endl;
    std::cout.flags(original_flags);
}

class tests{
public:
    tests(std::size_t size):size_(size) { }

    template <class TestType>
    void runTest(const TestType &t, std::size_t min_duration) {
        test<TestType> tst(t, min_duration);
        elapsed.push_back( tst.execute());
        names.push_back(t.name());
    }

    void print_results() {
        std::ios_base::fmtflags original_flags = std::cout.flags();
        std::cout <<  std::setw(6) << size_ << ',' << std::setfill(' ')<<  std::setprecision(4) << std::setw(names[0].size()+2) << std::right <<4*mflops(size_, elapsed[0]);
        for (std::size_t i=1; i!=elapsed.size(); i++)
            std::cout << ',' << std::setfill(' ')<<  std::setprecision(4) << std::setw(names[i].size()+2) << std::right << 4*mflops(size_, elapsed[i]);
        std::cout << std::endl;
        std::cout.flags(original_flags);
    }

    void print_header() {
        std::ios_base::fmtflags original_flags = std::cout.flags();
        std::cout <<  std::setw(6) << "Size" << ',' << std::setfill(' ')<< std::setw(names[0].size()+2) << std::right << names[0];
        for (std::size_t i=1; i!=elapsed.size(); i++)
            std::cout << ',' << std::setfill(' ')<<  std::setw(names[i].size()+2) << std::right <<names[i];
        std::cout << std::endl;
        std::cout.flags(original_flags);
    }

private:
    std::size_t                 size_;
    std::vector<double>         elapsed;
    std::vector<std::string>    names;
};


int main(int /*argc*/, char **/*argv*/) {


    std::cout << std::numeric_limits<std::size_t>::max() << std::endl;
    omp_set_num_threads(2);

    std::cout << eigen3(256).sample_output() << ", " << ublas_base(256).sample_output()<< ", " << ublas_noalias(256).sample_output()<< ", " << ublas_axpy(256).sample_output() <<  " " << ublas_axpy2(256).sample_output() << " " << ublas_paxpy(256, 4).sample_output() << std::endl;

    std::size_t start = 5;
    std::size_t end =   14;
    for (std::size_t size= COMPUTE_STRIDE_SIZE; size<1200; size+=COMPUTE_STRIDE_SIZE) {
  //  for (std::size_t p=start; p!=end; p++) {
   //     std::size_t size = std::pow( 2, p );
        tests ts(size);
        ts.runTest(eigen3(size), 100000) ;
        ts.runTest(ublas_base(size), 100000) ;
        ts.runTest(ublas_noalias(size), 100000) ;
        ts.runTest(ublas_axpy(size), 100000) ;
        ts.runTest(ublas_axpy2(size), 100000) ;
        for (int i=1; i<= omp_get_num_procs()/2; i++) {

            ts.runTest(ublas_paxpy(size, i), 100000);
        }
        if ( size == COMPUTE_STRIDE_SIZE ) ts.print_header();
        ts.print_results();
    }
    std::size_t size = 10000;
    tests ts(size);
    ts.runTest(eigen3(size), 100000) ;
    ts.runTest(ublas_base(size), 100000) ;
    ts.runTest(ublas_noalias(size), 100000) ;
    ts.runTest(ublas_axpy(size), 100000) ;
    ts.runTest(ublas_axpy2(size), 100000) ;
    for (int i=1; i<= omp_get_num_procs()/2; i++) {

        ts.runTest(ublas_paxpy(size, i), 100000);
    }
    ts.print_results();

    //    std::vector<double> elapsed;
/*
        elapsed.push_back( runTest(eigen3(size), 100000) );
        elapsed.push_back( runTest(ublas_base(size), 100000) );
        elapsed.push_back( runTest(ublas_axpy(size), 100000) );
        elapsed.push_back( runTest(ublas_axpy2(size), 100000) );
        for (std::size_t i=1; i<= omp_get_num_procs(); i++) {
            omp_set_num_threads(i);
            elapsed.push_back( runTest(ublas_paxpy(size), 100000) );
        }*/

        //print_results(size, elapsed);
   // }





    return 0;
}

