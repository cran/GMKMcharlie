// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// d2s
List d2s(NumericMatrix X, double zero, double threshold, bool verbose);
RcppExport SEXP _GMKMcharlie_d2s(SEXP XSEXP, SEXP zeroSEXP, SEXP thresholdSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< double >::type zero(zeroSEXP);
    Rcpp::traits::input_parameter< double >::type threshold(thresholdSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(d2s(X, zero, threshold, verbose));
    return rcpp_result_gen;
END_RCPP
}
// s2d
NumericMatrix s2d(List X, int d, double zero, bool verbose);
RcppExport SEXP _GMKMcharlie_s2d(SEXP XSEXP, SEXP dSEXP, SEXP zeroSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    Rcpp::traits::input_parameter< double >::type zero(zeroSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(s2d(X, d, zero, verbose));
    return rcpp_result_gen;
END_RCPP
}
// testGdensity
double testGdensity(NumericVector x, NumericVector mu, NumericVector sigma, double alpha);
RcppExport SEXP _GMKMcharlie_testGdensity(SEXP xSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(testGdensity(x, mu, sigma, alpha));
    return rcpp_result_gen;
END_RCPP
}
// findSpreadedMeanWrapper
NumericMatrix findSpreadedMeanWrapper(NumericMatrix X, int K, int maxCore);
RcppExport SEXP _GMKMcharlie_findSpreadedMeanWrapper(SEXP XSEXP, SEXP KSEXP, SEXP maxCoreSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< int >::type maxCore(maxCoreSEXP);
    rcpp_result_gen = Rcpp::wrap(findSpreadedMeanWrapper(X, K, maxCore));
    return rcpp_result_gen;
END_RCPP
}
// makeCovariancesWrapper
NumericMatrix makeCovariancesWrapper(NumericMatrix X, int K);
RcppExport SEXP _GMKMcharlie_makeCovariancesWrapper(SEXP XSEXP, SEXP KSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    rcpp_result_gen = Rcpp::wrap(makeCovariancesWrapper(X, K));
    return rcpp_result_gen;
END_RCPP
}
// paraGmm
List paraGmm(NumericMatrix X, NumericVector Xw, int G, NumericVector alpha, NumericMatrix mu, NumericMatrix sigma, double eigenRatioLim, double convergenceEPS, double alphaEPS, int maxIter, double tlimit, int verbose, int maxCore);
RcppExport SEXP _GMKMcharlie_paraGmm(SEXP XSEXP, SEXP XwSEXP, SEXP GSEXP, SEXP alphaSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP eigenRatioLimSEXP, SEXP convergenceEPSSEXP, SEXP alphaEPSSEXP, SEXP maxIterSEXP, SEXP tlimitSEXP, SEXP verboseSEXP, SEXP maxCoreSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Xw(XwSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type eigenRatioLim(eigenRatioLimSEXP);
    Rcpp::traits::input_parameter< double >::type convergenceEPS(convergenceEPSSEXP);
    Rcpp::traits::input_parameter< double >::type alphaEPS(alphaEPSSEXP);
    Rcpp::traits::input_parameter< int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< double >::type tlimit(tlimitSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type maxCore(maxCoreSEXP);
    rcpp_result_gen = Rcpp::wrap(paraGmm(X, Xw, G, alpha, mu, sigma, eigenRatioLim, convergenceEPS, alphaEPS, maxIter, tlimit, verbose, maxCore));
    return rcpp_result_gen;
END_RCPP
}
// paraGmmCW
List paraGmmCW(NumericMatrix X, NumericVector Xw, int G, NumericVector alpha, NumericMatrix mu, NumericMatrix sigma, double eigenRatioLim, double convergenceEPS, double alphaEPS, int maxIter, double tlimit, int verbose, int maxCore);
RcppExport SEXP _GMKMcharlie_paraGmmCW(SEXP XSEXP, SEXP XwSEXP, SEXP GSEXP, SEXP alphaSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP eigenRatioLimSEXP, SEXP convergenceEPSSEXP, SEXP alphaEPSSEXP, SEXP maxIterSEXP, SEXP tlimitSEXP, SEXP verboseSEXP, SEXP maxCoreSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Xw(XwSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type eigenRatioLim(eigenRatioLimSEXP);
    Rcpp::traits::input_parameter< double >::type convergenceEPS(convergenceEPSSEXP);
    Rcpp::traits::input_parameter< double >::type alphaEPS(alphaEPSSEXP);
    Rcpp::traits::input_parameter< int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< double >::type tlimit(tlimitSEXP);
    Rcpp::traits::input_parameter< int >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type maxCore(maxCoreSEXP);
    rcpp_result_gen = Rcpp::wrap(paraGmmCW(X, Xw, G, alpha, mu, sigma, eigenRatioLim, convergenceEPS, alphaEPS, maxIter, tlimit, verbose, maxCore));
    return rcpp_result_gen;
END_RCPP
}
// paraGmmFJ
List paraGmmFJ(NumericMatrix X, NumericVector Xw, int G, int Gmin, NumericVector alpha, NumericMatrix mu, NumericMatrix sigma, double eigenRatioLim, double convergenceEPS, double alphaEPS, int maxIter, double tlimit, bool verbose, int maxCore);
RcppExport SEXP _GMKMcharlie_paraGmmFJ(SEXP XSEXP, SEXP XwSEXP, SEXP GSEXP, SEXP GminSEXP, SEXP alphaSEXP, SEXP muSEXP, SEXP sigmaSEXP, SEXP eigenRatioLimSEXP, SEXP convergenceEPSSEXP, SEXP alphaEPSSEXP, SEXP maxIterSEXP, SEXP tlimitSEXP, SEXP verboseSEXP, SEXP maxCoreSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Xw(XwSEXP);
    Rcpp::traits::input_parameter< int >::type G(GSEXP);
    Rcpp::traits::input_parameter< int >::type Gmin(GminSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type sigma(sigmaSEXP);
    Rcpp::traits::input_parameter< double >::type eigenRatioLim(eigenRatioLimSEXP);
    Rcpp::traits::input_parameter< double >::type convergenceEPS(convergenceEPSSEXP);
    Rcpp::traits::input_parameter< double >::type alphaEPS(alphaEPSSEXP);
    Rcpp::traits::input_parameter< int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< double >::type tlimit(tlimitSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< int >::type maxCore(maxCoreSEXP);
    rcpp_result_gen = Rcpp::wrap(paraGmmFJ(X, Xw, G, Gmin, alpha, mu, sigma, eigenRatioLim, convergenceEPS, alphaEPS, maxIter, tlimit, verbose, maxCore));
    return rcpp_result_gen;
END_RCPP
}
// KMcpp
List KMcpp(NumericMatrix X, NumericMatrix centroid, NumericVector Xw, double minkP, int maxCore, int maxIter, bool verbose);
RcppExport SEXP _GMKMcharlie_KMcpp(SEXP XSEXP, SEXP centroidSEXP, SEXP XwSEXP, SEXP minkPSEXP, SEXP maxCoreSEXP, SEXP maxIterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type centroid(centroidSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Xw(XwSEXP);
    Rcpp::traits::input_parameter< double >::type minkP(minkPSEXP);
    Rcpp::traits::input_parameter< int >::type maxCore(maxCoreSEXP);
    Rcpp::traits::input_parameter< int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(KMcpp(X, centroid, Xw, minkP, maxCore, maxIter, verbose));
    return rcpp_result_gen;
END_RCPP
}
// KMconstrainedCpp
List KMconstrainedCpp(NumericMatrix X, NumericMatrix centroids, NumericVector Xw, NumericVector clusterWeightUpperBound, double minkP, int maxCore, int convergenceTail, double tailConvergedRelaErr, int maxIter, bool paraSortInplaceMerge, bool verbose);
RcppExport SEXP _GMKMcharlie_KMconstrainedCpp(SEXP XSEXP, SEXP centroidsSEXP, SEXP XwSEXP, SEXP clusterWeightUpperBoundSEXP, SEXP minkPSEXP, SEXP maxCoreSEXP, SEXP convergenceTailSEXP, SEXP tailConvergedRelaErrSEXP, SEXP maxIterSEXP, SEXP paraSortInplaceMergeSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type centroids(centroidsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Xw(XwSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type clusterWeightUpperBound(clusterWeightUpperBoundSEXP);
    Rcpp::traits::input_parameter< double >::type minkP(minkPSEXP);
    Rcpp::traits::input_parameter< int >::type maxCore(maxCoreSEXP);
    Rcpp::traits::input_parameter< int >::type convergenceTail(convergenceTailSEXP);
    Rcpp::traits::input_parameter< double >::type tailConvergedRelaErr(tailConvergedRelaErrSEXP);
    Rcpp::traits::input_parameter< int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type paraSortInplaceMerge(paraSortInplaceMergeSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(KMconstrainedCpp(X, centroids, Xw, clusterWeightUpperBound, minkP, maxCore, convergenceTail, tailConvergedRelaErr, maxIter, paraSortInplaceMerge, verbose));
    return rcpp_result_gen;
END_RCPP
}
// KMppIniCpp
IntegerVector KMppIniCpp(NumericMatrix X, int firstSelection, int K, double minkP, bool stochastic, double seed, int maxCore, bool verbose);
RcppExport SEXP _GMKMcharlie_KMppIniCpp(SEXP XSEXP, SEXP firstSelectionSEXP, SEXP KSEXP, SEXP minkPSEXP, SEXP stochasticSEXP, SEXP seedSEXP, SEXP maxCoreSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericMatrix >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type firstSelection(firstSelectionSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< double >::type minkP(minkPSEXP);
    Rcpp::traits::input_parameter< bool >::type stochastic(stochasticSEXP);
    Rcpp::traits::input_parameter< double >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type maxCore(maxCoreSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(KMppIniCpp(X, firstSelection, K, minkP, stochastic, seed, maxCore, verbose));
    return rcpp_result_gen;
END_RCPP
}
// KMppIniSparseCpp
IntegerVector KMppIniSparseCpp(List X, int d, int firstSelection, int K, double minkP, bool stochastic, double seed, int maxCore, bool verbose);
RcppExport SEXP _GMKMcharlie_KMppIniSparseCpp(SEXP XSEXP, SEXP dSEXP, SEXP firstSelectionSEXP, SEXP KSEXP, SEXP minkPSEXP, SEXP stochasticSEXP, SEXP seedSEXP, SEXP maxCoreSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    Rcpp::traits::input_parameter< int >::type firstSelection(firstSelectionSEXP);
    Rcpp::traits::input_parameter< int >::type K(KSEXP);
    Rcpp::traits::input_parameter< double >::type minkP(minkPSEXP);
    Rcpp::traits::input_parameter< bool >::type stochastic(stochasticSEXP);
    Rcpp::traits::input_parameter< double >::type seed(seedSEXP);
    Rcpp::traits::input_parameter< int >::type maxCore(maxCoreSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(KMppIniSparseCpp(X, d, firstSelection, K, minkP, stochastic, seed, maxCore, verbose));
    return rcpp_result_gen;
END_RCPP
}
// sparseKMcpp
List sparseKMcpp(List X, int d, List centroid, NumericVector Xw, double minkP, int maxCore, int maxIter, bool verbose);
RcppExport SEXP _GMKMcharlie_sparseKMcpp(SEXP XSEXP, SEXP dSEXP, SEXP centroidSEXP, SEXP XwSEXP, SEXP minkPSEXP, SEXP maxCoreSEXP, SEXP maxIterSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    Rcpp::traits::input_parameter< List >::type centroid(centroidSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Xw(XwSEXP);
    Rcpp::traits::input_parameter< double >::type minkP(minkPSEXP);
    Rcpp::traits::input_parameter< int >::type maxCore(maxCoreSEXP);
    Rcpp::traits::input_parameter< int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(sparseKMcpp(X, d, centroid, Xw, minkP, maxCore, maxIter, verbose));
    return rcpp_result_gen;
END_RCPP
}
// sparseKMconstrainedCpp
List sparseKMconstrainedCpp(List X, int d, List centroid, NumericVector Xw, NumericVector clusterWeightUpperBound, double minkP, int maxCore, int convergenceTail, double tailConvergedRelaErr, int maxIter, bool paraSortInplaceMerge, bool verbose);
RcppExport SEXP _GMKMcharlie_sparseKMconstrainedCpp(SEXP XSEXP, SEXP dSEXP, SEXP centroidSEXP, SEXP XwSEXP, SEXP clusterWeightUpperBoundSEXP, SEXP minkPSEXP, SEXP maxCoreSEXP, SEXP convergenceTailSEXP, SEXP tailConvergedRelaErrSEXP, SEXP maxIterSEXP, SEXP paraSortInplaceMergeSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< List >::type X(XSEXP);
    Rcpp::traits::input_parameter< int >::type d(dSEXP);
    Rcpp::traits::input_parameter< List >::type centroid(centroidSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type Xw(XwSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type clusterWeightUpperBound(clusterWeightUpperBoundSEXP);
    Rcpp::traits::input_parameter< double >::type minkP(minkPSEXP);
    Rcpp::traits::input_parameter< int >::type maxCore(maxCoreSEXP);
    Rcpp::traits::input_parameter< int >::type convergenceTail(convergenceTailSEXP);
    Rcpp::traits::input_parameter< double >::type tailConvergedRelaErr(tailConvergedRelaErrSEXP);
    Rcpp::traits::input_parameter< int >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< bool >::type paraSortInplaceMerge(paraSortInplaceMergeSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    rcpp_result_gen = Rcpp::wrap(sparseKMconstrainedCpp(X, d, centroid, Xw, clusterWeightUpperBound, minkP, maxCore, convergenceTail, tailConvergedRelaErr, maxIter, paraSortInplaceMerge, verbose));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_GMKMcharlie_d2s", (DL_FUNC) &_GMKMcharlie_d2s, 4},
    {"_GMKMcharlie_s2d", (DL_FUNC) &_GMKMcharlie_s2d, 4},
    {"_GMKMcharlie_testGdensity", (DL_FUNC) &_GMKMcharlie_testGdensity, 4},
    {"_GMKMcharlie_findSpreadedMeanWrapper", (DL_FUNC) &_GMKMcharlie_findSpreadedMeanWrapper, 3},
    {"_GMKMcharlie_makeCovariancesWrapper", (DL_FUNC) &_GMKMcharlie_makeCovariancesWrapper, 2},
    {"_GMKMcharlie_paraGmm", (DL_FUNC) &_GMKMcharlie_paraGmm, 13},
    {"_GMKMcharlie_paraGmmCW", (DL_FUNC) &_GMKMcharlie_paraGmmCW, 13},
    {"_GMKMcharlie_paraGmmFJ", (DL_FUNC) &_GMKMcharlie_paraGmmFJ, 14},
    {"_GMKMcharlie_KMcpp", (DL_FUNC) &_GMKMcharlie_KMcpp, 7},
    {"_GMKMcharlie_KMconstrainedCpp", (DL_FUNC) &_GMKMcharlie_KMconstrainedCpp, 11},
    {"_GMKMcharlie_KMppIniCpp", (DL_FUNC) &_GMKMcharlie_KMppIniCpp, 8},
    {"_GMKMcharlie_KMppIniSparseCpp", (DL_FUNC) &_GMKMcharlie_KMppIniSparseCpp, 9},
    {"_GMKMcharlie_sparseKMcpp", (DL_FUNC) &_GMKMcharlie_sparseKMcpp, 8},
    {"_GMKMcharlie_sparseKMconstrainedCpp", (DL_FUNC) &_GMKMcharlie_sparseKMconstrainedCpp, 12},
    {NULL, NULL, 0}
};

RcppExport void R_init_GMKMcharlie(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}