#pragma once

#include <vector>

#include <opencv2/core.hpp>

#include "DM_Core.h"
#include "DM_IO.h"
#include "DM_ClusterSummary.h"

namespace dms {

    enum class RansacMode {
        DDD,  // dense matches
        CCC,  // cluster centers (no summarization)
        CCA   // cluster centers + summarized M_k
    };

    struct RansacOptions {
        int    numIterations = 1000;   // RANSAC iterations
        int    sampleSize = 8;     // 8-point algorithm
        // Threshold specified in pixels. Internally converted to normalized
        // coordinates via avg focal length (fx) from intrinsics.
        double inlierThresholdPx = 2.5;  // ScanNet pixel threshold set in the paper
    };

    struct MethodMetrics {
        double auc5 = 0.0;  // AUC up to 5 deg
        double auc10 = 0.0;  // AUC up to 10 deg
        double auc20 = 0.0;  // AUC up to 20 deg
        double avgRuntimeMs = 0.0;  // average over repeats
        double speedupWrtDDD = 1.0; // filled once DDD baseline is known
    };

    class RansacEvaluator {
    public:
        RansacEvaluator(const DatasetConfig& cfg,
            const RansacOptions& opts);

        // Evaluate DDD: dense matches (no clustering needed)
        MethodMetrics evaluateDDD(const MatchIO& matchIO,
            const DenseMatcher& dense,
            int numRepeats = 10) const;

        // Evaluate CCC: cluster centers (no summarization used)
        MethodMetrics evaluateCCC(const MatchIO& matchIO,
            const std::vector<ClusteredPair>& clustered,
            int numRepeats = 10) const;

        // Evaluate CCA: cluster centers + summarized M_k
        MethodMetrics evaluateCCA(const MatchIO& matchIO,
            const std::vector<ClusteredPair>& clustered,
            int numRepeats = 10) const;

    private:
        DatasetConfig cfg_;
        RansacOptions opts_;

        // Core RANSAC routine for one pair (dense or centers).
        // If summaries non-null, the final refinement uses cluster
        // summaries (CCA). Otherwise, refinement uses 8-point with inliers.
        // All computations done in *normalized* coordinates:
        //   x_norm  = K1^{-1} x_pixel
        //   x'_norm = K2^{-1} x'_pixel
        PoseResult runRansacGeneric(const std::vector<Match2D>& matchesPixel,
            const std::vector<ClusterSummary>* summaries,
            const cv::Matx33d& K1,
            const cv::Matx33d& K2) const;

        PoseResult runRansacDense(const MatchList& matches,
            const cv::Matx33d& K1,
            const cv::Matx33d& K2) const;

        PoseResult runRansacCenters(const std::vector<Match2D>& centers,
            const cv::Matx33d& K1,
            const cv::Matx33d& K2) const;

        PoseResult runRansacCCA(const std::vector<Match2D>& centers,
            const std::vector<ClusterSummary>& summaries,
            const cv::Matx33d& K1,
            const cv::Matx33d& K2) const;

        // Essential estimation (normalized 8-point)
        // Input points assumed to be in *normalized* camera coordinates.
        bool estimateEssential8Point(const std::vector<Match2D>& matches,
            const std::vector<int>& indices,
            cv::Matx33d& E_out) const;

        // Decompose essential matrix into R, t (up to sign)
        bool decomposeEssential(const cv::Matx33d& E,
            cv::Matx33d& R_out,
            cv::Vec3d& t_out) const;

        // Sampson error in normalized coordinates (scale threshold using fx).
        double sampsonError(const cv::Matx33d& E,
            const Match2D& m) const;

        // Pose error vs GT (max(rotError, transError) in degrees)
        PoseError computePoseError(const GroundTruthPose& gt,
            const PoseResult& est) const;

        // AUC helper (0–1 range)
        static void computeAucMetrics(
            const std::vector<PoseError>& perSampleErrors,
            double& auc5, double& auc10, double& auc20);

        // 2D normalization helper for 8-point algorithm
        static void normalizePoints2D(const std::vector<cv::Point2d>& pts,
            std::vector<cv::Point2d>& ptsNorm,
            cv::Matx33d& T);
    };

} // namespace dms
