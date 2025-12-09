#pragma once

#include <vector>
#include <random>
#include <opencv2/core.hpp>

#include "DM_Core.h"

namespace dms {

    // -----------------------------------------------------------------------------
    // ClusterSummary
    //   - One entry per cluster
    //   - Contains:
    //       * representative match coordinates (c1, c2)
    //       * representative score
    //       * cluster size
    //       * 9x9 matrix M_k = sum_i (a_i a_i^T) for all matches in this cluster
    //         where a_i is the 9D epipolar constraint vector in *normalized*
    //         camera coordinates (based on the paper):
    //             x_norm  = K1^{-1} [x,  y,  1]^T
    //             x'_norm = K2^{-1} [x', y', 1]^T
    //         and a_i = vec(x'_norm x_norm^T).
    // -----------------------------------------------------------------------------
    struct ClusterSummary
    {
        cv::Point2f c1;        // representative point in image 1 (pixels)
        cv::Point2f c2;        // representative point in image 2 (pixels)
        float       score = 0; // score of representative match
        int         size = 0; // number of dense matches in this cluster

        // 9x9 summarized correspondence matrix for the cluster
        cv::Matx<double, 9, 9> M;

        ClusterSummary()
            : c1(0.f, 0.f), c2(0.f, 0.f), score(0.f), size(0),
            M(cv::Matx<double, 9, 9>::zeros())
        {
        }
    };

    // -----------------------------------------------------------------------------
    // ClusteredPair
    //   - Result for one image pair:
    //       * pointer back to dense matches (non-owning)
    //       * cluster label for each dense match
    //       * summary info per cluster
    // -----------------------------------------------------------------------------
    struct ClusteredPair
    {
        int pairIndex = -1;

        // Non-owning pointer to dense matches for this pair.
        // Assumed point into DenseMatcher's internal storage.
        const MatchList* matches = nullptr;

        // labels[i] == cluster index for matches->at(i), in [0, clusters.size()-1]
        std::vector<int> labels;

        // One summary per cluster
        std::vector<ClusterSummary> clusters;
    };

    // -----------------------------------------------------------------------------
    // MatchClusterer
    //   - 4D K-means clustering on (x, y, x', y').
    //   - K is usually 128 as in the paper.
    //   - For each pair, returns ClusteredPair with:
    //       * assignments
    //       * representative matches
    //       * precomputed 9x9 M_k summary matrices (in normalized coords).
    // -----------------------------------------------------------------------------
    class MatchClusterer
    {
    public:
        // numClusters: desired K (will be clamped to [1, N] per pair)
        // maxIterations: K-means iterations (paper used up to 5)
        explicit MatchClusterer(int numClusters = 128,
            int maxIterations = 5);

        int numClusters()   const { return numClusters_; }
        int maxIterations() const { return maxIterations_; }

        // Cluster a single pair's dense matches.
        // - pairIndex for debug prints.
        // - matches must remain alive as long as ClusteredPair is used
        //   (store only a pointer here, not a copy).
        // - K1, K2 are intrinsics for the *color* cameras of this pair.
        ClusteredPair clusterPair(int pairIndex,
            const MatchList& matches,
            const cv::Matx33d& K1,
            const cv::Matx33d& K2) const;

    private:
        int numClusters_;
        int maxIterations_;
        mutable std::mt19937 rng_;

        // 4D feature (x, y, x', y') in pixel coordinates for clustering.
        static cv::Vec4d makeFeature(const Match2D& m);

        // Run basic Lloyd K-means in 4D.
        // Inputs:
        //   features: N 4D points
        //   K:       number of clusters
        // Outputs:
        //   labels:  size N, label i in [0, K-1]
        //   centers: size K, center vectors
        void runKMeans4D(const std::vector<cv::Vec4d>& features,
            int K,
            std::vector<int>& labels,
            std::vector<cv::Vec4d>& centers) const;

        // Build summary for a single cluster (representative + M_k)
        ClusterSummary buildClusterSummary(int clusterIndex,
            const MatchList& matches,
            const std::vector<int>& labels,
            const cv::Vec4d& center,
            const cv::Matx33d& K1_inv,
            const cv::Matx33d& K2_inv) const;

        // Compute M_k = sum_{i in cluster k} a_i a_i^T
        // where a_i is based on *normalized* coordinates.
        static cv::Matx<double, 9, 9> computeMForCluster(
            int clusterIndex,
            const MatchList& matches,
            const std::vector<int>& labels,
            const cv::Matx33d& K1_inv,
            const cv::Matx33d& K2_inv);

        // Build 9D epipolar constraint vector a = vec(x'_norm x_norm^T)
        // using normalized image coords:
        //   x_norm  = K1^{-1} [x,  y,  1]^T
        //   x'_norm = K2^{-1} [x', y', 1]^T
        static cv::Matx<double, 9, 1> makeConstraintVec(
            const cv::Point2f& p1,
            const cv::Point2f& p2,
            const cv::Matx33d& K1_inv,
            const cv::Matx33d& K2_inv);
    };

} // namespace dms
