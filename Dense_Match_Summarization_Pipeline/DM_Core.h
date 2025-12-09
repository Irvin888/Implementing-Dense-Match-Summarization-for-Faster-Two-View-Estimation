#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace dms {

    //======================================================================
    // Dataset + matcher configuration
    //======================================================================

    enum class DatasetType {
        MegaDepth1500, // not used
        ScanNet1500
    };

    enum class DenseMatcherType {
        PrecomputedFiles   // use precomputed matches
    };

    struct DatasetConfig {
        DatasetType      dataset = DatasetType::ScanNet1500;
        DenseMatcherType matcherType = DenseMatcherType::PrecomputedFiles;

        // Original dataset root.
        // Potentially useful for:
        //   - image visualization
        //   - fallback for intrinsics/poses if precomputed data is missing
        std::string datasetRoot;
        std::string imageRoot1;
        std::string imageRoot2;

        // Precomputed outputs (from the Python script)
        //   matches:   precomputed_dkm/matches/sceneXXXX_00_000000_000045.txt
        //   gt_pose:   precomputed_dkm/gt_pose/sceneXXXX_00_000000_000045_pose.txt
        //   intrinsics: precomputed_dkm/intrinsics/sceneXXXX_00_intrinsics.txt
        std::string matchOutputRoot;        // "precomputed_dkm/matches"
        std::string precomputedGtPoseRoot;  // "precomputed_dkm/gt_pose"
        std::string precomputedIntrinsicsRoot; // "precomputed_dkm/intrinsics"

        // Global limits to keep experiments small
        int maxScenesToLoad = -1;  // -1 = all scenes
        int maxPairsPerScene = -1;  // -1 = all pairs in each scene
        int numPairsToProcess = -1;  // -1 = use all discovered pairs

        int defaultNumClusters = 128; // K for clustering (CCC/CCA)

    };

    //======================================================================
    // Core data types
    //======================================================================

    struct Match2D {
        cv::Point2f p1;   // image 1
        cv::Point2f p2;   // image 2
        float       score = 0.f;
    };

    using MatchList = std::vector<Match2D>;

    struct ImagePair {
        std::string sceneId;   // e.g. "scene0742_00"
        int frameIdx1 = -1;
        int frameIdx2 = -1;

        std::string imgPath1;  // color image 1
        std::string imgPath2;  // color image 2

        // Original ScanNet pose txt files (not used for now)
        std::string posePath1;
        std::string posePath2;
    };

    struct GroundTruthPose {
        cv::Matx33d R_gt;  // rotation
        cv::Vec3d   t_gt;  // unit-length translation direction
    };

    struct PoseResult {
        cv::Matx33d R;
        cv::Vec3d   t;
        bool        success = false;
    };

    struct PoseError {
        double rotErrorDeg = 0.0;
        double transErrorDeg = 0.0;
        double maxErrorDeg = 0.0;
    };

} // namespace dms
