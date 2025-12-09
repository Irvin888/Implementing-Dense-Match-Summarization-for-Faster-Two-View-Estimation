#include "DM_IO.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace dms {

    //============================================================
    // MatchIO
    //============================================================

    MatchIO::MatchIO(const DatasetConfig& cfg)
        : cfg_(cfg)
    {
    }

    std::string MatchIO::pad6(int idx)
    {
        char buf[16];
        std::snprintf(buf, sizeof(buf), "%06d", idx);
        return std::string(buf);
    }

    bool MatchIO::loadImagePairs()
    {
        pairs_.clear();

        if (cfg_.dataset != DatasetType::ScanNet1500) {
            std::cerr << "[MatchIO] Only ScanNet1500 is supported.\n";
            return false;
        }

        if (cfg_.datasetRoot.empty()) {
            std::cerr << "[MatchIO] datasetRoot is empty.\n";
            return false;
        }

        fs::path root(cfg_.datasetRoot);

        if (!fs::exists(root) || !fs::is_directory(root)) {
            std::cerr << "[MatchIO] datasetRoot does not exist or is not a directory: "
                << root << "\n";
            return false;
        }

        // Collect scene directories
        std::vector<fs::path> sceneDirs;
        for (const auto& entry : fs::directory_iterator(root)) {
            if (entry.is_directory()) {
                sceneDirs.push_back(entry.path());
            }
        }
        std::sort(sceneDirs.begin(), sceneDirs.end(),
            [](const fs::path& a, const fs::path& b) {
                return a.filename().string() < b.filename().string();
            });

        if (sceneDirs.empty()) {
            std::cerr << "[MatchIO] No scene directories found under " << root << "\n";
            return false;
        }

        if (cfg_.maxScenesToLoad > 0 &&
            static_cast<int>(sceneDirs.size()) > cfg_.maxScenesToLoad)
        {
            sceneDirs.resize(cfg_.maxScenesToLoad);
        }

        int totalPairs = 0;
        bool reachedGlobalLimit = false;

        for (const auto& scenePath : sceneDirs) {
            if (reachedGlobalLimit) break;

            const std::string sceneId = scenePath.filename().string();
            fs::path colorDir = scenePath / "color";
            fs::path poseDir = scenePath / "pose";

            if (!fs::exists(colorDir) || !fs::is_directory(colorDir)) {
                std::cerr << "[MatchIO] Scene " << sceneId
                    << " missing color/, skipping.\n";
                continue;
            }
            if (!fs::exists(poseDir) || !fs::is_directory(poseDir)) {
                std::cerr << "[MatchIO] Scene " << sceneId
                    << " missing pose/, skipping.\n";
                continue;
            }

            // Collect numeric frame indices from color/
            std::vector<int> frames;
            for (const auto& entry : fs::directory_iterator(colorDir)) {
                if (!entry.is_regular_file()) continue;
                const auto stem = entry.path().stem().string();
                try {
                    int idx = std::stoi(stem);
                    frames.push_back(idx);
                }
                catch (...) {
                    // ignore non-numeric names
                }
            }
            std::sort(frames.begin(), frames.end());

            if (frames.size() < 2) {
                std::cerr << "[MatchIO] Scene " << sceneId
                    << " has fewer than 2 frames, skipping.\n";
                continue;
            }

            // Make sure even number of frames for pairing (0-1, 2-3, ...)
            if (frames.size() % 2 != 0) {
                frames.pop_back();
            }

            std::vector<std::pair<int, int>> scenePairs;
            for (size_t i = 0; i + 1 < frames.size(); i += 2) {
                scenePairs.emplace_back(frames[i], frames[i + 1]);
            }

            if (cfg_.maxPairsPerScene > 0 &&
                static_cast<int>(scenePairs.size()) > cfg_.maxPairsPerScene)
            {
                scenePairs.resize(cfg_.maxPairsPerScene);
            }

            if (scenePairs.empty()) {
                std::cerr << "[MatchIO] Scene " << sceneId
                    << " produced no frame pairs.\n";
                continue;
            }

            for (const auto& [f1, f2] : scenePairs) {
                ImagePair p;
                p.sceneId = sceneId;
                p.frameIdx1 = f1;
                p.frameIdx2 = f2;

                p.imgPath1 = (colorDir / (std::to_string(f1) + ".jpg")).string();
                p.imgPath2 = (colorDir / (std::to_string(f2) + ".jpg")).string();

                p.posePath1 = (poseDir / (std::to_string(f1) + ".txt")).string();
                p.posePath2 = (poseDir / (std::to_string(f2) + ".txt")).string();

                pairs_.push_back(std::move(p));
                ++totalPairs;

                if (cfg_.numPairsToProcess > 0 &&
                    totalPairs >= cfg_.numPairsToProcess)
                {
                    reachedGlobalLimit = true;
                    break;
                }
            }
        }

        if (pairs_.empty()) {
            std::cerr << "[MatchIO] No pairs found in dataset.\n";
            return false;
        }

        std::cout << "[MatchIO] Loaded " << totalPairs << " pair(s).\n";
        return true;
    }

    bool MatchIO::loadImagesForPair(int index, cv::Mat& img1, cv::Mat& img2) const
    {
        if (index < 0 || index >= numPairs()) {
            std::cerr << "[MatchIO] loadImagesForPair: index out of range.\n";
            return false;
        }

        const ImagePair& p = pairs_[index];
        img1 = cv::imread(p.imgPath1, cv::IMREAD_COLOR);
        img2 = cv::imread(p.imgPath2, cv::IMREAD_COLOR);

        if (img1.empty()) {
            std::cerr << "[MatchIO] Failed to load image: " << p.imgPath1 << "\n";
            return false;
        }
        if (img2.empty()) {
            std::cerr << "[MatchIO] Failed to load image: " << p.imgPath2 << "\n";
            return false;
        }
        return true;
    }

    bool MatchIO::loadGroundTruthPoseForPair(
        int index,
        GroundTruthPose& outPose,
        bool usePrecomputed
    ) const
    {
        if (index < 0 || index >= numPairs()) {
            std::cerr << "[MatchIO] loadGroundTruthPoseForPair: index out of range.\n";
            return false;
        }

        const ImagePair& p = pairs_[index];

        // First choice: use precomputed GT pose from precomputed_dkm/gt_pose
        if (usePrecomputed && !cfg_.precomputedGtPoseRoot.empty()) {
            fs::path root(cfg_.precomputedGtPoseRoot);
            const std::string baseName =
                p.sceneId + "_" + pad6(p.frameIdx1) + "_" + pad6(p.frameIdx2) + "_pose.txt";
            fs::path poseFile = root / baseName;

            if (!fs::exists(poseFile)) {
                return false;
            }

            std::ifstream in(poseFile.string());
            if (!in) {
                std::cerr << "[MatchIO] Failed to open precomputed GT pose: "
                    << poseFile << "\n";
                return false;
            }

            double vals[12];
            for (int i = 0; i < 12; ++i) {
                if (!(in >> vals[i])) {
                    std::cerr << "[MatchIO] Failed to read 12 numbers from: "
                        << poseFile << "\n";
                    return false;
                }
            }

            outPose.R_gt = cv::Matx33d(
                vals[0], vals[1], vals[2],
                vals[3], vals[4], vals[5],
                vals[6], vals[7], vals[8]
            );
            outPose.t_gt = cv::Vec3d(vals[9], vals[10], vals[11]);
            return true;
        }

        // If above failed: compute from original ScanNet pose/ files
        auto loadPoseMatrix = [](const std::string& path, cv::Matx44d& T) -> bool {
            std::ifstream in(path);
            if (!in) {
                std::cerr << "[MatchIO] Failed to open pose file: " << path << "\n";
                return false;
            }
            double vals[16];
            for (int i = 0; i < 16; ++i) {
                if (!(in >> vals[i])) {
                    std::cerr << "[MatchIO] Pose file has fewer than 16 numbers: "
                        << path << "\n";
                    return false;
                }
            }
            T = cv::Matx44d(
                vals[0], vals[1], vals[2], vals[3],
                vals[4], vals[5], vals[6], vals[7],
                vals[8], vals[9], vals[10], vals[11],
                vals[12], vals[13], vals[14], vals[15]
            );
            return true;
            };

        cv::Matx44d T1, T2;
        if (!loadPoseMatrix(p.posePath1, T1) || !loadPoseMatrix(p.posePath2, T2)) {
            return false;
        }

        // ScanNet poses are Camera-to-World.
        // Need the transform FROM Cam1 TO Cam2.
        // P_world = T1 * P_c1
        // P_c2 = T2_inv * P_world
        // P_c2 = T2_inv * T1 * P_c1
        // So T_rel = T2.inv() * T1

        cv::Matx44d T2_inv = T2.inv();
        cv::Matx44d T_rel = T2_inv * T1;

        outPose.R_gt = cv::Matx33d(
            T_rel(0, 0), T_rel(0, 1), T_rel(0, 2),
            T_rel(1, 0), T_rel(1, 1), T_rel(1, 2),
            T_rel(2, 0), T_rel(2, 1), T_rel(2, 2)
        );
        outPose.t_gt = cv::Vec3d(T_rel(0, 3), T_rel(1, 3), T_rel(2, 3));

        double norm = cv::norm(outPose.t_gt);
        if (norm > 1e-9) {
            outPose.t_gt /= norm;
        }
        return true;
    }

    // ------------------------------------------------------------
    // Intrinsics loading (precomputed -> dataset fallback)
    // ------------------------------------------------------------

    bool MatchIO::loadIntrinsicsMatrixFromFile(
        const std::string& path,
        cv::Matx33d& K
    )
    {
        std::ifstream in(path);
        if (!in) {
            std::cerr << "[MatchIO] Failed to open intrinsics file: " << path << "\n";
            return false;
        }

        std::vector<double> vals;
        vals.reserve(16);
        double v;
        while (in >> v) {
            vals.push_back(v);
        }

        if (vals.size() != 9 && vals.size() != 16) {
            std::cerr << "[MatchIO] Intrinsics file " << path
                << " has " << vals.size()
                << " numbers (expected 9 or 16).\n";
            return false;
        }

        if (vals.size() == 9) {
            K = cv::Matx33d(
                vals[0], vals[1], vals[2],
                vals[3], vals[4], vals[5],
                vals[6], vals[7], vals[8]
            );
        }
        else { // 16
            // interpret as 4x4 row-major, take top-left 3x3
            K = cv::Matx33d(
                vals[0], vals[1], vals[2],
                vals[4], vals[5], vals[6],
                vals[8], vals[9], vals[10]
            );
        }

        return true;
    }

    bool MatchIO::loadIntrinsicsFromPrecomputed(
        const std::string& sceneId,
        cv::Matx33d& K
    ) const
    {
        if (cfg_.precomputedIntrinsicsRoot.empty()) {
            return false;
        }

        fs::path root(cfg_.precomputedIntrinsicsRoot);
        fs::path intrFile = root / (sceneId + "_intrinsics.txt");

        if (!fs::exists(intrFile)) {
            return false;
        }

        return loadIntrinsicsMatrixFromFile(intrFile.string(), K);
    }

    bool MatchIO::loadIntrinsicsFromDataset(
        const std::string& sceneId,
        cv::Matx33d& K
    ) const
    {
        if (cfg_.datasetRoot.empty()) {
            return false;
        }

        fs::path scenePath = fs::path(cfg_.datasetRoot) / sceneId;
        fs::path intrinsicDir = scenePath / "intrinsic";
        if (!fs::exists(intrinsicDir) || !fs::is_directory(intrinsicDir)) {
            return false;
        }

        // Try expected name first
        fs::path candidate = intrinsicDir / "intrinsic_color.txt";

        if (fs::exists(candidate) && fs::is_regular_file(candidate)) {
            return loadIntrinsicsMatrixFromFile(candidate.string(), K);
        }

        // Fallback: look for any file starting with "intrinsic_color"
        for (const auto& entry : fs::directory_iterator(intrinsicDir)) {
            if (!entry.is_regular_file()) continue;
            const std::string name = entry.path().filename().string();
            if (name.rfind("intrinsic_color", 0) == 0) {
                return loadIntrinsicsMatrixFromFile(entry.path().string(), K);
            }
        }

        return false;
    }

    bool MatchIO::loadIntrinsicsForPair(
        int index,
        cv::Matx33d& K1,
        cv::Matx33d& K2
    ) const
    {
        if (index < 0 || index >= numPairs()) {
            std::cerr << "[MatchIO] loadIntrinsicsForPair: index out of range.\n";
            return false;
        }

        const std::string& sceneId = pairs_[index].sceneId;
        cv::Matx33d K;

        // First try precomputed intrinsics
        if (loadIntrinsicsFromPrecomputed(sceneId, K)) {
            K1 = K;
            K2 = K;
            return true;
        }

        // Fallback: original dataset intrinsics
        if (loadIntrinsicsFromDataset(sceneId, K)) {
            K1 = K;
            K2 = K;
            return true;
        }

        std::cerr << "[MatchIO] Failed to load intrinsics for scene " << sceneId << "\n";
        return false;
    }

    //============================================================
    // DenseMatcher
    //============================================================

    DenseMatcher::DenseMatcher(const DatasetConfig& cfg)
        : cfg_(cfg)
    {
    }

    std::string DenseMatcher::matchFilePathForPair(const ImagePair& p) const
    {
        fs::path root(cfg_.matchOutputRoot);
        std::string baseName =
            p.sceneId + "_" +
            MatchIO::pad6(p.frameIdx1) + "_" +
            MatchIO::pad6(p.frameIdx2) + ".txt";
        fs::path full = root / baseName;
        return full.string();
    }

    bool DenseMatcher::loadMatchesFromFile(const std::string& path, MatchList& out) const
    {
        out.clear();
        std::ifstream in(path);
        if (!in) {
            std::cerr << "[DenseMatcher] Failed to open matches file: " << path << "\n";
            return false;
        }

        float x1, y1, x2, y2, score;
        int lineNo = 0;
        while (true) {
            if (!(in >> x1 >> y1 >> x2 >> y2 >> score)) {
                if (in.eof()) break;
                std::cerr << "[DenseMatcher] Parse error in " << path
                    << " at line " << (lineNo + 1) << "\n";
                return false;
            }
            Match2D m;
            m.p1 = cv::Point2f(x1, y1);
            m.p2 = cv::Point2f(x2, y2);
            m.score = score;
            out.push_back(m);
            ++lineNo;
        }
        return true;
    }

    bool DenseMatcher::computeOrLoadAllMatches(const MatchIO& matchIO)
    {
        const int n = matchIO.numPairs();
        matches_.assign(n, MatchList{});

        if (cfg_.matchOutputRoot.empty()) {
            std::cerr << "[DenseMatcher] matchOutputRoot is empty.\n";
            return false;
        }

        bool allOk = true;
        int loadedPairs = 0;
        std::size_t totalMatches = 0;

        for (int i = 0; i < n; ++i) {
            const ImagePair& p = matchIO.pair(i);
            const std::string path = matchFilePathForPair(p);

            fs::path fpath(path);
            if (!fs::exists(fpath)) {
                allOk = false;
                continue;
            }

            if (!loadMatchesFromFile(path, matches_[i])) {
                std::cerr << "[DenseMatcher] Failed to parse matches for pair "
                    << i << " from file " << path << "\n";
                allOk = false;
                continue;
            }

            ++loadedPairs;
            totalMatches += matches_[i].size();
        }

        if (loadedPairs > 0) {
            double avg = static_cast<double>(totalMatches) /
                static_cast<double>(loadedPairs);
            std::cout << "[DenseMatcher] Loaded matches for " << loadedPairs
                << " / " << n << " pairs (avg "
                << avg << " matches per loaded pair).\n";
        }
        else {
            std::cout << "[DenseMatcher] No matches loaded for any pair.\n";
        }

        return allOk;
    }


} // namespace dms
//////////////////