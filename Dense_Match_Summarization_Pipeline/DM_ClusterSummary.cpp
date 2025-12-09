#include "DM_ClusterSummary.h"

#include <algorithm>
#include <limits>
#include <numeric>

namespace dms {

    // -----------------------------------------------------------------------------
    // MatchClusterer implementation
    // -----------------------------------------------------------------------------

    MatchClusterer::MatchClusterer(int numClusters,
        int maxIterations)
        : numClusters_(std::max(1, numClusters)),
        maxIterations_(std::max(1, maxIterations)),
        rng_(123456u) // fixed seed for deterministic clustering
    {
    }

    cv::Vec4d MatchClusterer::makeFeature(const Match2D& m)
    {
        return cv::Vec4d(
            static_cast<double>(m.p1.x),
            static_cast<double>(m.p1.y),
            static_cast<double>(m.p2.x),
            static_cast<double>(m.p2.y)
        );
    }

    void MatchClusterer::runKMeans4D(const std::vector<cv::Vec4d>& features,
        int K,
        std::vector<int>& labels,
        std::vector<cv::Vec4d>& centers) const
    {
        const int N = static_cast<int>(features.size());
        if (N == 0 || K <= 0) {
            labels.assign(N, 0);
            centers.clear();
            return;
        }

        const int actualK = std::min(K, N);

        labels.assign(N, 0);
        centers.resize(actualK);

        // Initialization: choose centers by striding through the data
        int step = std::max(1, N / actualK);
        int idx = 0;
        for (int k = 0; k < actualK; ++k) {
            centers[k] = features[idx];
            idx = std::min(N - 1, idx + step);
        }

        // Temporary accumulators
        std::vector<cv::Vec4d> newCenters(actualK);
        std::vector<int>       counts(actualK);

        for (int iter = 0; iter < maxIterations_; ++iter) {
            bool changed = false;

            // Assignment step
            for (int i = 0; i < N; ++i) {
                const cv::Vec4d& x = features[i];

                double bestDist = std::numeric_limits<double>::max();
                int bestK = 0;
                for (int k = 0; k < actualK; ++k) {
                    cv::Vec4d diff = x - centers[k];
                    double d = diff.dot(diff);
                    if (d < bestDist) {
                        bestDist = d;
                        bestK = k;
                    }
                }

                if (labels[i] != bestK) {
                    labels[i] = bestK;
                    changed = true;
                }
            }

            // Recompute centers
            std::fill(newCenters.begin(), newCenters.end(),
                cv::Vec4d(0.0, 0.0, 0.0, 0.0));
            std::fill(counts.begin(), counts.end(), 0);

            for (int i = 0; i < N; ++i) {
                int k = labels[i];
                newCenters[k] += features[i];
                counts[k] += 1;
            }

            for (int k = 0; k < actualK; ++k) {
                if (counts[k] > 0) {
                    centers[k] = newCenters[k] * (1.0 / counts[k]);
                }
                else {
                    // Empty cluster: re-initialize randomly
                    int randomIndex = static_cast<int>(
                        rng_() % static_cast<unsigned>(N)
                        );
                    centers[k] = features[randomIndex];
                }
            }

            if (!changed) {
                break; // converged
            }
        }

        // labels are in [0, actualK-1]; caller uses centers.size() as K.
    }

    ClusterSummary MatchClusterer::buildClusterSummary(
        int clusterIndex,
        const MatchList& matches,
        const std::vector<int>& labels,
        const cv::Vec4d& center,
        const cv::Matx33d& K1_inv,
        const cv::Matx33d& K2_inv) const
    {
        const int N = static_cast<int>(matches.size());

        ClusterSummary summary;

        double bestDist = std::numeric_limits<double>::max();
        int representativeIdx = -1;
        int count = 0;

        // Find representative match closest to the center in 4D
        for (int i = 0; i < N; ++i) {
            if (labels[i] != clusterIndex) continue;
            ++count;

            cv::Vec4d feat = makeFeature(matches[i]);
            cv::Vec4d diff = feat - center;
            double d = diff.dot(diff);
            if (d < bestDist) {
                bestDist = d;
                representativeIdx = i;
            }
        }

        // Degenerate case: no points in cluster (should be rare after K-means).
        // Pick a representative to avoid crashes; M will be zero.
        if (count == 0) {
            representativeIdx = clusterIndex % std::max(1, N);
            count = 1;
        }

        const Match2D& rep = matches[representativeIdx];
        summary.c1 = rep.p1;
        summary.c2 = rep.p2;
        summary.score = rep.score;
        summary.size = count;

        // Compute M_k from all matches in this cluster (normalized coords)
        summary.M = computeMForCluster(clusterIndex, matches, labels,
            K1_inv, K2_inv);

        return summary;
    }

    cv::Matx<double, 9, 9> MatchClusterer::computeMForCluster(
        int clusterIndex,
        const MatchList& matches,
        const std::vector<int>& labels,
        const cv::Matx33d& K1_inv,
        const cv::Matx33d& K2_inv)
    {
        cv::Matx<double, 9, 9> M = cv::Matx<double, 9, 9>::zeros();

        const int N = static_cast<int>(matches.size());
        for (int i = 0; i < N; ++i) {
            if (labels[i] != clusterIndex) continue;

            const Match2D& m = matches[i];
            cv::Matx<double, 9, 1> a =
                makeConstraintVec(m.p1, m.p2, K1_inv, K2_inv);

            // Outer product a * a^T
            for (int r = 0; r < 9; ++r) {
                for (int c = 0; c < 9; ++c) {
                    M(r, c) += a(r, 0) * a(c, 0);
                }
            }
        }

        return M;
    }

    cv::Matx<double, 9, 1> MatchClusterer::makeConstraintVec(
        const cv::Point2f& p1,
        const cv::Point2f& p2,
        const cv::Matx33d& K1_inv,
        const cv::Matx33d& K2_inv)
    {
        // Pixel coordinates -> homogeneous
        cv::Vec3d x(p1.x, p1.y, 1.0);
        cv::Vec3d xp(p2.x, p2.y, 1.0);

        // Normalize by intrinsics: x_norm = K1^{-1} x, x'_norm = K2^{-1} x'
        cv::Vec3d xN = K1_inv * x;
        cv::Vec3d xpN = K2_inv * xp;

        // In case the third component is not 1, homogenize.
        if (std::abs(xN[2]) > 1e-12) {
            xN[0] /= xN[2];
            xN[1] /= xN[2];
            xN[2] = 1.0;
        }
        if (std::abs(xpN[2]) > 1e-12) {
            xpN[0] /= xpN[2];
            xpN[1] /= xpN[2];
            xpN[2] = 1.0;
        }

        const double x_norm = xN[0];
        const double y_norm = xN[1];
        const double xp_norm = xpN[0];
        const double yp_norm = xpN[1];

        // Standard x'_norm ⊗ x_norm for the essential matrix constraint:
        //
        //  [xp * x, xp * y, xp,
        //   yp * x, yp * y, yp,
        //   x,      y,      1]
        //
        return cv::Matx<double, 9, 1>(
            xp_norm * x_norm,
            xp_norm * y_norm,
            xp_norm,
            yp_norm * x_norm,
            yp_norm * y_norm,
            yp_norm,
            x_norm,
            y_norm,
            1.0
        );
    }

    ClusteredPair MatchClusterer::clusterPair(
        int pairIndex,
        const MatchList& matches,
        const cv::Matx33d& K1,
        const cv::Matx33d& K2) const
    {
        ClusteredPair result;
        result.pairIndex = pairIndex;
        result.matches = &matches;

        const int N = static_cast<int>(matches.size());
        if (N == 0) {
            return result; // empty pair
        }

        // Build 4D features in pixel space
        std::vector<cv::Vec4d> features;
        features.reserve(N);
        for (const Match2D& m : matches) {
            features.push_back(makeFeature(m));
        }

        // Run K-means in 4D
        std::vector<int> labels;
        std::vector<cv::Vec4d> centers;
        runKMeans4D(features, numClusters_, labels, centers);

        const int K = static_cast<int>(centers.size());

        result.labels = std::move(labels);
        result.clusters.clear();
        result.clusters.reserve(K);

        // Precompute K^{-1} once for this pair
        cv::Matx33d K1_inv = K1.inv();
        cv::Matx33d K2_inv = K2.inv();

        for (int k = 0; k < K; ++k) {
            ClusterSummary summary =
                buildClusterSummary(k, matches, result.labels,
                    centers[k], K1_inv, K2_inv);
            result.clusters.push_back(summary);
        }

        return result;
    }

} // namespace dms
