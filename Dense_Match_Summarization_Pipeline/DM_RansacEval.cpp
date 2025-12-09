#include "DM_RansacEval.h"

#include <random>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <opencv2/calib3d.hpp>

namespace dms {

    using Clock = std::chrono::high_resolution_clock;

    RansacEvaluator::RansacEvaluator(const DatasetConfig& cfg,
        const RansacOptions& opts)
        : cfg_(cfg),
        opts_(opts)
    {
    }

    // ---------------------------------------------------------------
    // Utility: normalize 2D points
    // ---------------------------------------------------------------
    void RansacEvaluator::normalizePoints2D(const std::vector<cv::Point2d>& pts,
        std::vector<cv::Point2d>& ptsNorm,
        cv::Matx33d& T)
    {
        const int N = static_cast<int>(pts.size());
        ptsNorm.clear();
        ptsNorm.reserve(N);

        if (N == 0) {
            T = cv::Matx33d::eye();
            return;
        }

        double meanX = 0.0, meanY = 0.0;
        for (const auto& p : pts) {
            meanX += p.x;
            meanY += p.y;
        }
        meanX /= N;
        meanY /= N;

        double meanDist = 0.0;
        for (const auto& p : pts) {
            double dx = p.x - meanX;
            double dy = p.y - meanY;
            meanDist += std::sqrt(dx * dx + dy * dy);
        }
        meanDist /= N;

        double s = (meanDist > 1e-9) ? (std::sqrt(2.0) / meanDist) : 1.0;

        T = cv::Matx33d(
            s, 0, -s * meanX,
            0, s, -s * meanY,
            0, 0, 1
        );

        for (const auto& p : pts) {
            cv::Vec3d v(p.x, p.y, 1.0);
            cv::Vec3d vn = T * v;
            ptsNorm.emplace_back(vn[0] / vn[2], vn[1] / vn[2]);
        }
    }

    // ---------------------------------------------------------------
    // Estimate essential matrix via normalized 8-point algorithm
    // ---------------------------------------------------------------
    bool RansacEvaluator::estimateEssential8Point(const std::vector<Match2D>& matches,
        const std::vector<int>& indices,
        cv::Matx33d& E_out) const
    {
        const int n = static_cast<int>(indices.size());
        if (n < 8) {
            return false;
        }

        // Build constraint matrix A such that A*e = 0, where e = vec(E)
        cv::Mat A(n, 9, CV_64F);
        for (int i = 0; i < n; ++i) {
            const Match2D& m = matches[indices[i]];

            // Points are already in normalized coordinates
            double x = m.p1.x;
            double y = m.p1.y;
            double xp = m.p2.x;
            double yp = m.p2.y;

            double* row = A.ptr<double>(i);
            // For x'^T E x = 0, (x' ⊗ x)^T vec(E) = 0
            row[0] = xp * x;
            row[1] = xp * y;
            row[2] = xp;
            row[3] = yp * x;
            row[4] = yp * y;
            row[5] = yp;
            row[6] = x;
            row[7] = y;
            row[8] = 1.0;
        }

        // Solve via SVD: find nullspace of A
        cv::Mat w, u, vt;
        cv::SVD::compute(A, w, u, vt, cv::SVD::FULL_UV);

        // Get solution from smallest singular value (last row of V^T)
        cv::Mat eVec = vt.row(vt.rows - 1);

        // Reshape to 3x3 matrix
        cv::Mat E_init(3, 3, CV_64F);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                E_init.at<double>(r, c) = eVec.at<double>(0, 3 * r + c);
            }
        }

        // Enforce essential matrix constraints: rank 2 with two equal singular values
        cv::Mat wE, uE, vtE;
        cv::SVD::compute(E_init, wE, uE, vtE, cv::SVD::FULL_UV);

        double s0 = wE.at<double>(0, 0);
        double s1 = wE.at<double>(1, 0);

        // Check for degeneracy
        if (s0 < 1e-6) {
            return false;
        }

        // Project to essential matrix manifold
        double s_avg = 0.5 * (s0 + s1);

        cv::Mat S = cv::Mat::zeros(3, 3, CV_64F);
        S.at<double>(0, 0) = s_avg;
        S.at<double>(1, 1) = s_avg;
        S.at<double>(2, 2) = 0.0;

        cv::Mat E_final = uE * S * vtE;

        // Convert to Matx33d
        E_out = cv::Matx33d(
            E_final.at<double>(0, 0), E_final.at<double>(0, 1), E_final.at<double>(0, 2),
            E_final.at<double>(1, 0), E_final.at<double>(1, 1), E_final.at<double>(1, 2),
            E_final.at<double>(2, 0), E_final.at<double>(2, 1), E_final.at<double>(2, 2)
        );

        return true;
    }

    // ---------------------------------------------------------------
    // Decompose essential matrix into R, t (up to sign)
    // ---------------------------------------------------------------
    bool RansacEvaluator::decomposeEssential(const cv::Matx33d& E,
        cv::Matx33d& R_out,
        cv::Vec3d& t_out) const
    {
        std::cout << "[DEBUG decompose] Starting essential decomposition\n";

        cv::Mat Em(3, 3, CV_64F);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                Em.at<double>(r, c) = E(r, c);

        cv::Mat w, u, vt;
        try {
            cv::SVD::compute(Em, w, u, vt, cv::SVD::FULL_UV);
        }
        catch (const cv::Exception& e) {
            std::cerr << "[decompose] SVD failed: " << e.what() << "\n";
            return false;
        }

        std::cout << "[DEBUG decompose] E decomposition singular values: "
            << w.at<double>(0) << ", "
            << w.at<double>(1) << ", "
            << w.at<double>(2) << "\n";

        // Ensure proper orientation
        if (cv::determinant(u) < 0) {
            u.col(2) *= -1;
        }
        if (cv::determinant(vt) < 0) {
            vt.row(2) *= -1;
        }

        cv::Mat W = (cv::Mat_<double>(3, 3) <<
            0, -1, 0,
            1, 0, 0,
            0, 0, 1);

        cv::Mat Rm = u * W * vt;
        double detR = cv::determinant(Rm);
        std::cout << "[DEBUG decompose] R determinant: " << detR << "\n";

        if (detR < 0) {
            Rm = -Rm;
        }

        cv::Mat tMat = u.col(2);
        cv::Vec3d t(tMat.at<double>(0), tMat.at<double>(1), tMat.at<double>(2));

        double normT = cv::norm(t);
        std::cout << "[DEBUG decompose] t norm before normalization: " << normT << "\n";

        if (normT > 1e-9) {
            t /= normT;
        }

        R_out = cv::Matx33d(
            Rm.at<double>(0, 0), Rm.at<double>(0, 1), Rm.at<double>(0, 2),
            Rm.at<double>(1, 0), Rm.at<double>(1, 1), Rm.at<double>(1, 2),
            Rm.at<double>(2, 0), Rm.at<double>(2, 1), Rm.at<double>(2, 2)
        );
        t_out = t;

        std::cout << "[DEBUG decompose] Successfully decomposed E\n";
        return true;
    }

    // ---------------------------------------------------------------
    // Sampson error (approx residual) in normalized coordinates
    // ---------------------------------------------------------------
    double RansacEvaluator::sampsonError(const cv::Matx33d& E,
        const Match2D& m) const
    {
        cv::Vec3d x(m.p1.x, m.p1.y, 1.0);
        cv::Vec3d xp(m.p2.x, m.p2.y, 1.0);

        cv::Vec3d Ex = E * x;
        cv::Vec3d Etxp = E.t() * xp;

        double xpEx = xp.dot(Ex);

        // Sampson error denominator: ||Ex||_2^2 + ||E^T xp||_2^2
        // where ||.||_2 considers only first two components
        double denom = Ex[0] * Ex[0] + Ex[1] * Ex[1] +
            Etxp[0] * Etxp[0] + Etxp[1] * Etxp[1];

        if (denom < 1e-12) {
            return 1e9;  // Invalid
        }

        // Sampson error: |xp^T E x| / sqrt(denom)
        double d = (xpEx * xpEx) / denom;
        return std::sqrt(d);
    }

    // ---------------------------------------------------------------
    // Pose error: rotation + translation angle in degrees
    // ---------------------------------------------------------------
    PoseError RansacEvaluator::computePoseError(const GroundTruthPose& gt,
        const PoseResult& est) const
    {
        PoseError err;
        if (!est.success) {
            err.rotErrorDeg = 180.0;
            err.transErrorDeg = 180.0;
            err.maxErrorDeg = 180.0;
            return err;
        }

        // Rotation error
        cv::Matx33d R_rel = est.R * gt.R_gt.t();
        double trace = R_rel(0, 0) + R_rel(1, 1) + R_rel(2, 2);
        double cosTheta = 0.5 * (trace - 1.0);
        cosTheta = std::max(-1.0, std::min(1.0, cosTheta));
        double thetaRad = std::acos(cosTheta);
        err.rotErrorDeg = thetaRad * 180.0 / CV_PI;

        // Translation direction error (sign ambiguous)
        cv::Vec3d t_est = est.t;
        cv::Vec3d t_gt = gt.t_gt;

        double nEst = cv::norm(t_est);
        double nGt = cv::norm(t_gt);
        if (nEst < 1e-9 || nGt < 1e-9) {
            err.transErrorDeg = 180.0;
        }
        else {
            t_est /= nEst;
            t_gt /= nGt;

            double dot = t_est.dot(t_gt);
            dot = std::max(-1.0, std::min(1.0, dot));
            dot = std::abs(dot); // ignore sign ambiguity
            double angRad = std::acos(dot);
            err.transErrorDeg = angRad * 180.0 / CV_PI;
        }

        err.maxErrorDeg = std::max(err.rotErrorDeg, err.transErrorDeg);
        return err;
    }

    // ---------------------------------------------------------------
    // Compute AUC@5,10,20 given per-sample errors (max rot/trans)
    // AUC@tau = average over samples of (max(0, tau - e) / tau)
    // ---------------------------------------------------------------
    void RansacEvaluator::computeAucMetrics(
        const std::vector<PoseError>& perSampleErrors,
        double& auc5, double& auc10, double& auc20)
    {
        auc5 = auc10 = auc20 = 0.0;
        if (perSampleErrors.empty()) {
            return;
        }

        auto computeForTau = [&](double tauDeg) {
            double sum = 0.0;
            for (const auto& e : perSampleErrors) {
                double m = e.maxErrorDeg;
                if (m < tauDeg) {
                    sum += (tauDeg - m) / tauDeg;
                }
            }
            return sum / static_cast<double>(perSampleErrors.size());
            };

        auc5 = computeForTau(5.0);
        auc10 = computeForTau(10.0);
        auc20 = computeForTau(20.0);
    }

    // ---------------------------------------------------------------
    // Core RANSAC (DDD/CCC/CCA)
    // All using *normalized* coordinates x_norm = K^{-1} x_pixel.
    // ---------------------------------------------------------------
    PoseResult RansacEvaluator::runRansacGeneric(
        const std::vector<Match2D>& matchesPixel,
        const std::vector<ClusterSummary>* summaries,
        const cv::Matx33d& K1,
        const cv::Matx33d& K2) const
    {
        PoseResult result;
        result.success = false;

        const int N = static_cast<int>(matchesPixel.size());
        if (N < opts_.sampleSize) {
            std::cerr << "[RANSAC] Too few matches: " << N << " < " << opts_.sampleSize << "\n";
            return result;
        }

        // Precompute normalized coordinates
        cv::Matx33d K1_inv = K1.inv();
        cv::Matx33d K2_inv = K2.inv();

        std::vector<Match2D> matchesNorm;
        matchesNorm.reserve(N);
        for (const Match2D& mp : matchesPixel) {
            cv::Vec3d x(mp.p1.x, mp.p1.y, 1.0);
            cv::Vec3d xp(mp.p2.x, mp.p2.y, 1.0);

            cv::Vec3d xN = K1_inv * x;
            cv::Vec3d xpN = K2_inv * xp;

            // Normalize to homogeneous coords with z=1
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

            Match2D mn;
            mn.p1 = cv::Point2f(static_cast<float>(xN[0]),
                static_cast<float>(xN[1]));
            mn.p2 = cv::Point2f(static_cast<float>(xpN[0]),
                static_cast<float>(xpN[1]));
            mn.score = mp.score;
            matchesNorm.push_back(mn);
        }

        // Convert pixel threshold to normalized coordinates properly
        // The Sampson error is in normalized coordinates, so need to scale by focal length
        double fx1 = K1(0, 0);
        double fy1 = K1(1, 1);
        double fx2 = K2(0, 0);
        double fy2 = K2(1, 1);

        // SANITY CHECK: Print some normalized coordinates
        if (N > 0) {
            std::cout << "[DEBUG] Sample normalized coordinates (first match):\n";
            std::cout << "  Pixel: (" << matchesPixel[0].p1.x << ", " << matchesPixel[0].p1.y
                << ") -> (" << matchesPixel[0].p2.x << ", " << matchesPixel[0].p2.y << ")\n";
            std::cout << "  Norm:  (" << matchesNorm[0].p1.x << ", " << matchesNorm[0].p1.y
                << ") -> (" << matchesNorm[0].p2.x << ", " << matchesNorm[0].p2.y << ")\n";

            // Normalized coords should typically be in range [-2, 2]
            if (std::abs(matchesNorm[0].p1.x) > 10 || std::abs(matchesNorm[0].p1.y) > 10) {
                std::cerr << "[WARNING] Normalized coordinates seem too large! Check intrinsics.\n";
            }
        }

        // Use average of all focal lengths
        double fAvg = (fx1 + fy1 + fx2 + fy2) / 4.0;
        if (fAvg <= 0.0) fAvg = 1.0;

        // Scale threshold: normalized_threshold = pixel_threshold / focal_length
        double inlierThrNorm = opts_.inlierThresholdPx / fAvg;

        std::cout << "[DEBUG] Pixel threshold: " << opts_.inlierThresholdPx
            << ", Normalized threshold: " << inlierThrNorm
            << ", Avg focal: " << fAvg << "\n";

        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_int_distribution<int> uni(0, N - 1);

        int bestInliers = 0;
        double bestResidualSum = std::numeric_limits<double>::max();
        cv::Matx33d bestE = cv::Matx33d::zeros();
        std::vector<int> bestInlierIndices;

        std::vector<int> sampleIdx(opts_.sampleSize);
        int validModels = 0;
        int totalInliersFound = 0;

        for (int it = 0; it < opts_.numIterations; ++it) {
            // Sample without replacement
            std::fill(sampleIdx.begin(), sampleIdx.end(), -1);
            int filled = 0;
            int attempts = 0;
            while (filled < opts_.sampleSize && attempts < N * 2) {
                int idx = uni(rng);
                bool used = false;
                for (int j = 0; j < filled; ++j) {
                    if (sampleIdx[j] == idx) { used = true; break; }
                }
                if (!used) {
                    sampleIdx[filled++] = idx;
                }
                attempts++;
            }

            if (filled < opts_.sampleSize) {
                std::cerr << "[RANSAC] Failed to sample " << opts_.sampleSize << " unique matches\n";
                continue;
            }

            cv::Matx33d E_candidate;
            if (!estimateEssential8Point(matchesNorm, sampleIdx, E_candidate)) {
                continue;
            }

            validModels++;

            int inlierCount = 0;
            double residualSum = 0.0;
            std::vector<int> inlierIndices;
            inlierIndices.reserve(N);

            for (int i = 0; i < N; ++i) {
                double err = sampsonError(E_candidate, matchesNorm[i]);
                if (err < inlierThrNorm) {
                    ++inlierCount;
                    residualSum += err;
                    inlierIndices.push_back(i);
                }
            }

            totalInliersFound = std::max(totalInliersFound, inlierCount);

            if (inlierCount > bestInliers ||
                (inlierCount == bestInliers && residualSum < bestResidualSum)) {
                bestInliers = inlierCount;
                bestResidualSum = residualSum;
                bestE = E_candidate;
                bestInlierIndices = std::move(inlierIndices);
            }
        }

        std::cout << "[DEBUG] Valid models generated: " << validModels << "/" << opts_.numIterations << "\n";
        std::cout << "[DEBUG] Best inlier count: " << bestInliers << "/" << N
            << " (" << (100.0 * bestInliers / N) << "%)\n";
        std::cout << "[DEBUG] Max inliers found in any iteration: " << totalInliersFound << "\n";

        if (bestInliers < opts_.sampleSize) {
            std::cerr << "[RANSAC] Too few inliers: " << bestInliers << " < " << opts_.sampleSize << "\n";
            return result;
        }

        if (bestInlierIndices.empty()) {
            std::cerr << "[RANSAC] No inlier indices stored\n";
            return result;
        }

        cv::Matx33d E_refined;

        if (summaries == nullptr) {
            // DDD / CCC: refine with 8-point over inliers
            std::cout << "[DEBUG] Refining with " << bestInlierIndices.size() << " inliers (DDD/CCC mode)\n";
            if (!estimateEssential8Point(matchesNorm, bestInlierIndices, E_refined)) {
                std::cerr << "[RANSAC] Refinement failed\n";
                return result;
            }
        }
        else {
            // CCA: use summarized cluster matrices
            std::cout << "[DEBUG] Refining with summarized M matrices (CCA mode), "
                << bestInlierIndices.size() << " inlier clusters\n";

            cv::Matx<double, 9, 9> M_agg = cv::Matx<double, 9, 9>::zeros();
            int validClusters = 0;

            for (int idx : bestInlierIndices) {
                if (idx < 0 || idx >= static_cast<int>(summaries->size())) {
                    std::cerr << "[RANSAC] Invalid cluster index: " << idx << "\n";
                    continue;
                }
                M_agg += (*summaries)[idx].M;
                validClusters++;
            }

            std::cout << "[DEBUG] Aggregated " << validClusters << " cluster M matrices\n";

            // Symmetrize to fix numerical errors
            for (int r = 0; r < 9; ++r) {
                for (int c = r + 1; c < 9; ++c) {
                    double avg = 0.5 * (M_agg(r, c) + M_agg(c, r));
                    M_agg(r, c) = avg;
                    M_agg(c, r) = avg;
                }
            }

            cv::Mat M(9, 9, CV_64F);
            for (int r = 0; r < 9; ++r) {
                for (int c = 0; c < 9; ++c) {
                    M.at<double>(r, c) = M_agg(r, c);
                }
            }

            cv::Mat eigenValues, eigenVectors;
            cv::eigen(M, eigenValues, eigenVectors);

            // Debug: check condition number
            double condNumber = eigenValues.at<double>(0) / std::max(1e-12, eigenValues.at<double>(8));
            std::cout << "[DEBUG] M matrix condition number: " << condNumber << "\n";
            std::cout << "[DEBUG] Smallest eigenvalue: " << eigenValues.at<double>(8) << "\n";

            // Get eigenvector for smallest eigenvalue (last row)
            cv::Mat eVec = eigenVectors.row(eigenVectors.rows - 1);

            cv::Mat E0(3, 3, CV_64F);
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    E0.at<double>(r, c) = eVec.at<double>(0, 3 * r + c);
                }
            }

            // Enforce essential matrix constraints
            cv::Mat wE, uE, vtE;
            cv::SVD::compute(E0, wE, uE, vtE, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

            double s1 = wE.at<double>(0);
            double s2 = wE.at<double>(1);
            double avg = 0.5 * (s1 + s2);

            cv::Mat S = cv::Mat::zeros(3, 3, CV_64F);
            S.at<double>(0, 0) = avg;
            S.at<double>(1, 1) = avg;
            // S(2,2) = 0 for essential matrix

            cv::Mat E_hat = uE * S * vtE;

            E_refined = cv::Matx33d(
                E_hat.at<double>(0, 0), E_hat.at<double>(0, 1), E_hat.at<double>(0, 2),
                E_hat.at<double>(1, 0), E_hat.at<double>(1, 1), E_hat.at<double>(1, 2),
                E_hat.at<double>(2, 0), E_hat.at<double>(2, 1), E_hat.at<double>(2, 2)
            );
        }

        // Try use cv::recoverPose to handle the 4-solution ambiguity (working for now)

        // 1. Gather the inlier points in normalized coordinates
        //    (need points to perform the Cheirality/Triangulation check)
        std::vector<cv::Point2d> inlierPts1, inlierPts2;
        if (bestInlierIndices.empty()) {
            // If running dense refinement without specific indices, use all matches
            inlierPts1.reserve(matchesNorm.size());
            inlierPts2.reserve(matchesNorm.size());
            for (const auto& m : matchesNorm) {
                inlierPts1.push_back(m.p1);
                inlierPts2.push_back(m.p2);
            }
        }
        else {
            inlierPts1.reserve(bestInlierIndices.size());
            inlierPts2.reserve(bestInlierIndices.size());
            for (int idx : bestInlierIndices) {
                if (idx >= 0 && idx < matchesNorm.size()) {
                    inlierPts1.push_back(matchesNorm[idx].p1);
                    inlierPts2.push_back(matchesNorm[idx].p2);
                }
            }
        }

        // 2. Convert E_refined to cv::Mat
        cv::Mat E_cv(E_refined);
        cv::Mat R_cv, t_cv;

        // 3. recoverPose automatically checks the 4 solutions and picks the one
        //    where points are in front of the camera.
        //    Expected points already normalized.
        int validPoints = cv::recoverPose(E_cv, inlierPts1, inlierPts2, R_cv, t_cv, 1.0, cv::Point2d(0, 0));

        if (validPoints <= 0) {
            std::cerr << "[RANSAC] recoverPose failed (Cheirality check failed)\n";
            return result;
        }

        // 4. Convert back to your Matx/Vec types
        cv::Matx33d R(reinterpret_cast<double*>(R_cv.data)); // Assumes CV_64F
        cv::Vec3d t(reinterpret_cast<double*>(t_cv.data));

        std::cout << "[DEBUG] Successfully estimated pose\n";
        result.R = R;
        result.t = t;
        result.success = true;
        return result;
    }

    PoseResult RansacEvaluator::runRansacDense(const MatchList& matches,
        const cv::Matx33d& K1,
        const cv::Matx33d& K2) const
    {
        return runRansacGeneric(matches, nullptr, K1, K2);
    }

    PoseResult RansacEvaluator::runRansacCenters(const std::vector<Match2D>& centers,
        const cv::Matx33d& K1,
        const cv::Matx33d& K2) const
    {
        return runRansacGeneric(centers, nullptr, K1, K2);
    }

    PoseResult RansacEvaluator::runRansacCCA(const std::vector<Match2D>& centers,
        const std::vector<ClusterSummary>& summaries,
        const cv::Matx33d& K1,
        const cv::Matx33d& K2) const
    {
        return runRansacGeneric(centers, &summaries, K1, K2);
    }

    // ---------------------------------------------------------------
    // Evaluation: DDD (dense matches)
    // ---------------------------------------------------------------
    MethodMetrics RansacEvaluator::evaluateDDD(const MatchIO& matchIO,
        const DenseMatcher& dense,
        int numRepeats) const
    {
        MethodMetrics metrics;

        const int numPairs = matchIO.numPairs();
        if (numPairs <= 0) return metrics;

        std::vector<PoseError> allErrors;
        allErrors.reserve(numPairs * numRepeats);

        double totalTimeMs = 0.0;

        for (int rep = 0; rep < numRepeats; ++rep) {
            auto t0 = Clock::now();

            for (int i = 0; i < numPairs; ++i) {
                GroundTruthPose gt;
                if (!matchIO.loadGroundTruthPoseForPair(i, gt, /*usePrecomputed=*/true) &&
                    !matchIO.loadGroundTruthPoseForPair(i, gt, /*usePrecomputed=*/false)) {
                    continue;
                }

                // In evaluateDDD, after loading GT:
                std::cout << "[DEBUG] GT pose for pair " << i << ":\n";
                std::cout << "  R_gt trace: " << (gt.R_gt(0, 0) + gt.R_gt(1, 1) + gt.R_gt(2, 2)) << " (should be ~3)\n";
                std::cout << "  t_gt norm: " << cv::norm(gt.t_gt) << " (should be ~1)\n";
                std::cout << "  R_gt det: " << cv::determinant(cv::Mat(3, 3, CV_64F, (void*)gt.R_gt.val)) << " (should be ~1)\n";

                cv::Matx33d K1, K2;
                if (!matchIO.loadIntrinsicsForPair(i, K1, K2)) {
                    std::cerr << "[RANSAC] DDD: missing intrinsics for pair " << i
                        << ", skipping.\n";
                    continue;
                }

                const MatchList& matches = dense.matchesForPair(i);
                if (matches.size() < static_cast<size_t>(opts_.sampleSize)) continue;

                PoseResult est = runRansacDense(matches, K1, K2);
                if (!est.success) continue;

                PoseError err = computePoseError(gt, est);
                allErrors.push_back(err);
            }

            auto t1 = Clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
            totalTimeMs += ms;
        }

        if (!allErrors.empty()) {
            computeAucMetrics(allErrors, metrics.auc5, metrics.auc10, metrics.auc20);
        }

        metrics.avgRuntimeMs = totalTimeMs / std::max(1, numRepeats);
        return metrics;
    }

    // ---------------------------------------------------------------
    // Build mapping from pair index -> ClusteredPair*
    // ---------------------------------------------------------------
    static std::vector<const ClusteredPair*>
        buildClusterMap(const MatchIO& matchIO,
            const std::vector<ClusteredPair>& clustered)
    {
        const int numPairs = matchIO.numPairs();
        std::vector<const ClusteredPair*> map(numPairs, nullptr);
        for (const auto& cp : clustered) {
            int idx = cp.pairIndex;
            if (idx >= 0 && idx < numPairs) {
                map[idx] = &cp;
            }
        }
        return map;
    }

    // ---------------------------------------------------------------
    // Evaluation: CCC (cluster centers only)
    // ---------------------------------------------------------------
    MethodMetrics RansacEvaluator::evaluateCCC(const MatchIO& matchIO,
        const std::vector<ClusteredPair>& clustered,
        int numRepeats) const
    {
        MethodMetrics metrics;

        const int numPairs = matchIO.numPairs();
        if (numPairs <= 0) return metrics;

        auto clusterMap = buildClusterMap(matchIO, clustered);

        std::vector<PoseError> allErrors;
        allErrors.reserve(numPairs * numRepeats);

        double totalTimeMs = 0.0;

        for (int rep = 0; rep < numRepeats; ++rep) {
            auto t0 = Clock::now();

            for (int i = 0; i < numPairs; ++i) {
                const ClusteredPair* cp = clusterMap[i];
                if (!cp || cp->clusters.empty()) continue;

                GroundTruthPose gt;
                if (!matchIO.loadGroundTruthPoseForPair(i, gt, /*usePrecomputed=*/true) &&
                    !matchIO.loadGroundTruthPoseForPair(i, gt, /*usePrecomputed=*/false)) {
                    continue;
                }

                cv::Matx33d K1, K2;
                if (!matchIO.loadIntrinsicsForPair(i, K1, K2)) {
                    std::cerr << "[RANSAC] CCC: missing intrinsics for pair " << i
                        << ", skipping.\n";
                    continue;
                }

                // build centers
                std::vector<Match2D> centers;
                centers.reserve(cp->clusters.size());
                for (const auto& cs : cp->clusters) {
                    Match2D m;
                    m.p1 = cs.c1;
                    m.p2 = cs.c2;
                    m.score = cs.score;
                    centers.push_back(m);
                }

                if (centers.size() < static_cast<size_t>(opts_.sampleSize)) continue;

                PoseResult est = runRansacCenters(centers, K1, K2);
                if (!est.success) continue;

                PoseError err = computePoseError(gt, est);
                allErrors.push_back(err);
            }

            auto t1 = Clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
            totalTimeMs += ms;
        }

        if (!allErrors.empty()) {
            computeAucMetrics(allErrors, metrics.auc5, metrics.auc10, metrics.auc20);
        }

        metrics.avgRuntimeMs = totalTimeMs / std::max(1, numRepeats);
        return metrics;
    }

    // ---------------------------------------------------------------
    // Evaluation: CCA (cluster centers + summarized M_k)
    // ---------------------------------------------------------------
    MethodMetrics RansacEvaluator::evaluateCCA(const MatchIO& matchIO,
        const std::vector<ClusteredPair>& clustered,
        int numRepeats) const
    {
        MethodMetrics metrics;

        const int numPairs = matchIO.numPairs();
        if (numPairs <= 0) return metrics;

        auto clusterMap = buildClusterMap(matchIO, clustered);

        std::vector<PoseError> allErrors;
        allErrors.reserve(numPairs * numRepeats);

        double totalTimeMs = 0.0;

        for (int rep = 0; rep < numRepeats; ++rep) {
            auto t0 = Clock::now();

            for (int i = 0; i < numPairs; ++i) {
                const ClusteredPair* cp = clusterMap[i];
                if (!cp || cp->clusters.empty()) continue;

                GroundTruthPose gt;
                if (!matchIO.loadGroundTruthPoseForPair(i, gt, /*usePrecomputed=*/true) &&
                    !matchIO.loadGroundTruthPoseForPair(i, gt, /*usePrecomputed=*/false)) {
                    continue;
                }

                cv::Matx33d K1, K2;
                if (!matchIO.loadIntrinsicsForPair(i, K1, K2)) {
                    std::cerr << "[RANSAC] CCA: missing intrinsics for pair " << i
                        << ", skipping.\n";
                    continue;
                }

                std::vector<Match2D> centers;
                centers.reserve(cp->clusters.size());
                for (const auto& cs : cp->clusters) {
                    Match2D m;
                    m.p1 = cs.c1;
                    m.p2 = cs.c2;
                    m.score = cs.score;
                    centers.push_back(m);
                }

                if (centers.size() < static_cast<size_t>(opts_.sampleSize)) continue;

                PoseResult est = runRansacCCA(centers, cp->clusters, K1, K2);
                if (!est.success) continue;

                PoseError err = computePoseError(gt, est);
                allErrors.push_back(err);
            }

            auto t1 = Clock::now();
            double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
            totalTimeMs += ms;
        }

        if (!allErrors.empty()) {
            computeAucMetrics(allErrors, metrics.auc5, metrics.auc10, metrics.auc20);
        }

        metrics.avgRuntimeMs = totalTimeMs / std::max(1, numRepeats);
        return metrics;
    }

} // namespace dms
