// Dense_Match_Summarization.cpp
//
// Stages:
//   1. Data loading + visualization
//   2. Clustering + summarization
//   3. RANSAC evaluation (DDD, CCC, CCA) with AUC@5/10/20, runtime, speedup
//
// Directory layout:
//
//   scannet_test_1500/
//     sceneXXXX_00/
//       color/
//       pose/
//
//   precomputed_dkm/
//     matches/     (dense DKM matches)
//     gt_pose/     (precomputed relative poses)
//     intrinsics/  (sceneXXXX_00_intrinsics.txt)
//
// And source files:
//   DM_Core.h
//   DM_IO.h / DM_IO.cpp
//   DM_ClusterSummary.h / DM_ClusterSummary.cpp
//   DM_RansacEval.h / DM_RansacEval.cpp
//
// Build: link with OpenCV (core, imgproc, highgui).

#include <iostream>
#include <vector>
#include <iomanip>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "DM_Core.h"
#include "DM_IO.h"
#include "DM_ClusterSummary.h"
#include "DM_RansacEval.h"

using namespace dms;

// -----------------------------------------------------------------------------
// Helper 1: data-loading + visualization
// -----------------------------------------------------------------------------
static void runDataLoadingAndVisualizationDemo(const DatasetConfig& cfg)
{
    MatchIO matchIO(cfg);

    if (!matchIO.loadImagePairs()) {
        std::cerr << "[Demo] ERROR: loadImagePairs() failed.\n";
        return;
    }

    const int numPairs = matchIO.numPairs();
    std::cout << "[Demo] Total pairs loaded: " << numPairs << "\n";

    // Load GT poses
    for (int i = 0; i < numPairs; ++i) {
        GroundTruthPose gt;
        if (!cfg.precomputedGtPoseRoot.empty() &&
            matchIO.loadGroundTruthPoseForPair(i, gt, /*usePrecomputed=*/true)) {
            std::cout << "[Demo] Pair " << i << ": GT pose loaded (precomputed).\n";
        }
        else if (matchIO.loadGroundTruthPoseForPair(i, gt, /*usePrecomputed=*/false)) {
            std::cout << "[Demo] Pair " << i << ": GT pose loaded (pose/ folder).\n";
        }
        else {
            std::cout << "[Demo] Pair " << i << ": FAILED to load GT pose.\n";
        }
    }

    // Load dense matches
    DenseMatcher dense(cfg);
    if (!dense.computeOrLoadAllMatches(matchIO)) {
        std::cout << "[Demo] WARNING: Some dense match files failed to load.\n";
    }

    if (numPairs == 0) {
        std::cerr << "[Demo] No pairs to visualize.\n";
        return;
    }

    cv::Mat img1, img2;
    if (!matchIO.loadImagesForPair(0, img1, img2)) {
        std::cerr << "[Demo] ERROR: Failed to load images for first pair.\n";
        return;
    }

    const MatchList& mlist = dense.matchesForPair(0);
    std::cout << "[Demo] First pair has " << mlist.size() << " dense matches.\n";

    cv::Mat vis;
    cv::hconcat(img1, img2, vis);

    const int kMaxMatchesToDraw = 1000;
    int maxToDraw = std::min<int>(static_cast<int>(mlist.size()),
        kMaxMatchesToDraw);

    for (int i = 0; i < maxToDraw; ++i) {
        const Match2D& m = mlist[i];

        cv::Point2f p1 = m.p1;
        cv::Point2f p2 = m.p2;
        p2.x += static_cast<float>(img1.cols);

        cv::circle(vis, p1, 2, cv::Scalar(0, 255, 0), -1);
        cv::circle(vis, p2, 2, cv::Scalar(0, 0, 255), -1);
        cv::line(vis, p1, p2, cv::Scalar(255, 0, 0), 1);
    }

    const std::string winName = "Dense matches (first pair)";
    cv::namedWindow(winName, cv::WINDOW_NORMAL);

    const int maxWidth = 1600;
    const int maxHeight = 900;

    cv::Mat visToShow = vis;
    if (vis.cols > maxWidth || vis.rows > maxHeight) {
        double sx = static_cast<double>(maxWidth) / vis.cols;
        double sy = static_cast<double>(maxHeight) / vis.rows;
        double scale = std::min(sx, sy);

        cv::Mat resized;
        cv::resize(vis, resized, cv::Size(), scale, scale, cv::INTER_AREA);
        visToShow = resized;
    }

    cv::imshow(winName, visToShow);
    std::cout << "[Demo] Showing " << maxToDraw << " matches. Press any key.\n";
    cv::waitKey(0);
}

// -----------------------------------------------------------------------------
// Helper 2: Clustering + summarization
// -----------------------------------------------------------------------------
static std::vector<ClusteredPair>
runClusteringAndSummarization(const DatasetConfig& cfg,
    const MatchIO& matchIO,
    const DenseMatcher& dense)
{
    std::cout << "\n[Cluster] Starting clustering + summarization...\n";

    const int numPairs = matchIO.numPairs();
    std::vector<ClusteredPair> clustered;
    clustered.reserve(numPairs);

    int K = (cfg.defaultNumClusters > 0) ? cfg.defaultNumClusters : 64;
    MatchClusterer clusterer(/*numClusters=*/K, /*maxIterations=*/5);

    int numClusteredPairs = 0;

    for (int i = 0; i < numPairs; ++i) {
        const MatchList& matches = dense.matchesForPair(i);
        if (matches.empty()) {
            continue;  // silent: no matches for this pair
        }

        cv::Matx33d K1, K2;
        if (!matchIO.loadIntrinsicsForPair(i, K1, K2)) {
            // Intrinsics missing for this pair: skip silently
            continue;
        }

        ClusteredPair cp = clusterer.clusterPair(i, matches, K1, K2);
        clustered.push_back(std::move(cp));
        ++numClusteredPairs;
    }

    if (clustered.empty()) {
        std::cout << "[Cluster] No clustered pairs produced.\n";
        return clustered;
    }

    std::cout << "[Cluster] Clustering finished for "
        << numClusteredPairs << " / " << numPairs
        << " pairs.\n";

    return clustered;

}

// -----------------------------------------------------------------------------
// Helper 3: RANSAC evaluation (DDD / CCC / CCA)
// -----------------------------------------------------------------------------
static void runRansacEvaluation(const DatasetConfig& cfg)
{
    std::cout << "\n[RANSAC] Starting evaluation (DDD / CCC / CCA)...\n";

    MatchIO matchIO(cfg);
    if (!matchIO.loadImagePairs()) {
        std::cerr << "[RANSAC] ERROR: loadImagePairs() failed.\n";
        return;
    }

    const int numPairs = matchIO.numPairs();
    if (numPairs == 0) {
        std::cerr << "[RANSAC] No image pairs to evaluate.\n";
        return;
    }

    DenseMatcher dense(cfg);
    if (!dense.computeOrLoadAllMatches(matchIO)) {
        std::cout << "[RANSAC] WARNING: Some dense match files failed to load.\n";
    }

    // After: dense.computeOrLoadAllMatches(matchIO)
    std::cout << "\n[DEBUG] Match statistics:\n";
    for (int i = 0; i < std::min(5, matchIO.numPairs()); ++i) {
        const MatchList& mlist = dense.matchesForPair(i);
        std::cout << "  Pair " << i << ": " << mlist.size() << " matches\n";

        if (!mlist.empty()) {
            cv::Matx33d K1, K2;
            if (matchIO.loadIntrinsicsForPair(i, K1, K2)) {
                std::cout << "    K1 fx=" << K1(0, 0) << ", fy=" << K1(1, 1)
                    << ", cx=" << K1(0, 2) << ", cy=" << K1(1, 2) << "\n";
            }
        }
    }

    // Cluster / summarization needed for CCC and CCA
    std::vector<ClusteredPair> clustered =
        runClusteringAndSummarization(cfg, matchIO, dense);

    if (clustered.empty()) {
        std::cerr << "[RANSAC] No clustered pairs, cannot run CCC/CCA.\n";
    }

    // RANSAC configuration
    RansacOptions ropts;
    ropts.numIterations = 750;   // tune as needed
    ropts.sampleSize = 8;
    ropts.inlierThresholdPx = 2.5;  // pixel threshold (converted to norm units inside)

    RansacEvaluator evaluator(cfg, ropts);

    const int numRepeats = 3;  // average runtime over a few runs

    MethodMetrics mDDD = evaluator.evaluateDDD(matchIO, dense, numRepeats);
    MethodMetrics mCCC, mCCA;

    if (!clustered.empty()) {
        mCCC = evaluator.evaluateCCC(matchIO, clustered, numRepeats);
        mCCA = evaluator.evaluateCCA(matchIO, clustered, numRepeats);
    }

    // Compute speedups w.r.t. DDD
    const double baselineTime = (mDDD.avgRuntimeMs > 1e-9)
        ? mDDD.avgRuntimeMs
        : 1.0;  // avoid div by zero

    mDDD.speedupWrtDDD = 1.0;
    mCCC.speedupWrtDDD = baselineTime / std::max(1e-9, mCCC.avgRuntimeMs);
    mCCA.speedupWrtDDD = baselineTime / std::max(1e-9, mCCA.avgRuntimeMs);

    // Print table similar to Table 4 in the paper
    std::cout << "\n[RANSAC] Results (averaged over " << numRepeats << " runs):\n\n";

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Method    "
        << "AUC@5   "
        << "AUC@10  "
        << "AUC@20  "
        << "Time(ms)  "
        << "Speedup\n";
    std::cout << "-------------------------------------------------------------\n";

    auto printRow = [](const char* name, const MethodMetrics& m) {
        std::cout << std::left << std::setw(8) << name << " "
            << std::setw(7) << m.auc5 << " "
            << std::setw(7) << m.auc10 << " "
            << std::setw(7) << m.auc20 << " "
            << std::setw(9) << m.avgRuntimeMs << " "
            << std::setw(0) << m.speedupWrtDDD << "x\n";
        };

    printRow("DDD", mDDD);
    printRow("CCC", mCCC);
    printRow("CCA", mCCA);

    std::cout << std::endl;
}

// -----------------------------------------------------------------------------
// main()
// -----------------------------------------------------------------------------
int main()
{
    // 1. Configure dataset
    DatasetConfig cfg;

    cfg.dataset = DatasetType::ScanNet1500;
    cfg.datasetRoot = "scannet_test_1500";
    cfg.imageRoot1 = cfg.datasetRoot;

    cfg.matchOutputRoot = "precomputed_dkm/matches";
    cfg.precomputedGtPoseRoot = "precomputed_dkm/gt_pose";
    cfg.precomputedIntrinsicsRoot = "precomputed_dkm/intrinsics";

    cfg.matcherType = DenseMatcherType::PrecomputedFiles;

    // For quick dev: restrict to a few scenes/pairs
    cfg.maxScenesToLoad = 5;   // e.g. 1 for quick test, -1 for all
    cfg.maxPairsPerScene = 6;   // e.g. 2 for quick test, -1 for all
    cfg.numPairsToProcess = 18;   // global cap across scenes, -1 = all
    cfg.defaultNumClusters = 128;

    // Choose which stages to run
    const bool kRunDataDemo = false; // visualize dense matches
    const bool kRunClusterPipeline = false; // cluster-only sanity
    const bool kRunRansacEval = true;  // full DDD/CCC/CCA evaluation

    if (kRunDataDemo) {
        runDataLoadingAndVisualizationDemo(cfg);
    }

    if (kRunRansacEval) {
        runRansacEvaluation(cfg);
    }
    else if (kRunClusterPipeline) {
        // Debug clustering info without RANSAC
        MatchIO matchIO(cfg);
        if (!matchIO.loadImagePairs()) {
            std::cerr << "[Main] ERROR: loadImagePairs() failed.\n";
            return 1;
        }
        DenseMatcher dense(cfg);
        if (!dense.computeOrLoadAllMatches(matchIO)) {
            std::cout << "[Main] WARNING: Some dense matches missing.\n";
        }
        (void)runClusteringAndSummarization(cfg, matchIO, dense);
    }

    return 0;
}
