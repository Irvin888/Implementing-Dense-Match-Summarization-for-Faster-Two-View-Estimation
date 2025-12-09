#pragma once

#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "DM_Core.h"

namespace dms {

    //============================================================
    // MatchIO: ScanNet images + GT pose + intrinsics loading
    //============================================================

    class MatchIO {
    public:
        explicit MatchIO(const DatasetConfig& cfg);

        // Discover image pairs.
        bool loadImagePairs();

        int numPairs() const { return static_cast<int>(pairs_.size()); }
        const ImagePair& pair(int index) const { return pairs_.at(index); }

        bool loadImagesForPair(int index, cv::Mat& img1, cv::Mat& img2) const;

        // Ground-truth relative pose for a pair:
        //   - usePrecomputed = true  -> try precomputed_dkm/gt_pose first
        //   - usePrecomputed = false -> fall back to ScanNet pose/ files
        bool loadGroundTruthPoseForPair(
            int index,
            GroundTruthPose& outPose,
            bool usePrecomputed = false
        ) const;

        // Intrinsics for a pair:
        //   - Expect precomputed_dkm/intrinsics/sceneId_intrinsics.txt
        //   - Fall back to sceneXXX_00/intrinsic/intrinsic_color[.txt]
        // For ScanNet color cameras, K1 == K2 for a given scene.
        bool loadIntrinsicsForPair(
            int index,
            cv::Matx33d& K1,
            cv::Matx33d& K2
        ) const;

        // helper: zero-pad index to 6 digits (public so DenseMatcher can use it)
        static std::string pad6(int idx);

    private:
        DatasetConfig          cfg_;
        std::vector<ImagePair> pairs_;

        // Helpers for intrinsics loading
        bool loadIntrinsicsFromPrecomputed(
            const std::string& sceneId,
            cv::Matx33d& K
        ) const;

        bool loadIntrinsicsFromDataset(
            const std::string& sceneId,
            cv::Matx33d& K
        ) const;

        static bool loadIntrinsicsMatrixFromFile(
            const std::string& path,
            cv::Matx33d& K
        );
    };

    //============================================================
    // DenseMatcher: load precomputed dense matches (DKM)
    //============================================================

    class DenseMatcher {
    public:
        explicit DenseMatcher(const DatasetConfig& cfg);

        // Load all precomputed matches for all pairs known by matchIO.
        bool computeOrLoadAllMatches(const MatchIO& matchIO);

        const MatchList& matchesForPair(int index) const {
            return matches_.at(index);
        }

    private:
        DatasetConfig          cfg_;
        std::vector<MatchList> matches_;

        std::string matchFilePathForPair(const ImagePair& p) const;
        bool        loadMatchesFromFile(const std::string& path, MatchList& out) const;
    };

} // namespace dms
