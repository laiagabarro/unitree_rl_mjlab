#include "State_PoseTracking.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"

State_PoseTracking::State_PoseTracking(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    // Get goal pose topic
    std::string goal_pose_topic = "rt/goal_pose";
    try
    {
        goal_pose_topic = cfg["goal_pose_topic"].as<std::string>();
    }
    catch (const std::exception &e)
    {
        // Use default topic
    }

    // Create goal pose subscription
    goal_pose_sub = std::make_shared<isaaclab::GoalPoseSubscription>(goal_pose_topic);

    // Create articulation with goal pose subscription
    auto articulation = std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(
        FSMState::lowstate, goal_pose_sub);

    // Enable debug prints
    try
    {
        debug_print = cfg["debug_print"].as<bool>();
    }
    catch (const std::exception &e)
    {
        debug_print = true; // Default to true for debugging 
    }

    // Load policy env config and optionally inject runtime overrides from controller config.yaml
    // NOTE: `env->cfg` comes from this policy `deploy.yaml` (not from param::config).
    YAML::Node env_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    {
        auto ranges = env_cfg["commands"]["pose_command"]["ranges"];

        // Inject PoseTracking-tuned parameters if present in controller config.yaml
        try
        {
            if (cfg["min_safe_distance"].IsDefined())
            {
                ranges["min_safe_distance"] = cfg["min_safe_distance"].as<float>();
            }
        }
        catch (const std::exception &e)
        {
        }

        try
        {
            if (cfg["yaw_offset"].IsDefined())
            {
                ranges["yaw_offset"] = cfg["yaw_offset"].as<float>();
            }
        }
        catch (const std::exception &e)
        {
        }

        // Ensure observation code can find debug_print in the pose_command ranges
        ranges["debug_print"] = debug_print;

        // Backward/forward compat: alias {pos_x,pos_y,heading} -> {x,y,yaw}
        // so the C++ observation implementation can clamp correctly.
        if (!ranges["x"].IsDefined() && ranges["pos_x"].IsDefined())
        {
            ranges["x"] = ranges["pos_x"];
        }
        if (!ranges["y"].IsDefined() && ranges["pos_y"].IsDefined())
        {
            ranges["y"] = ranges["pos_y"];
        }
        if (!ranges["yaw"].IsDefined() && ranges["heading"].IsDefined())
        {
            ranges["yaw"] = ranges["heading"];
        }
    }

    // Create environment
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        env_cfg,
        articulation);

    // Load ONNX policy
    auto onnx_path = policy_dir / "exported" / "policy.onnx";
    spdlog::info("Resolved ONNX path: {}", onnx_path.string());
    if (!std::filesystem::exists(onnx_path))
    {
        spdlog::critical("ONNX file does not exist: {}", onnx_path.string());
        throw std::runtime_error("Missing ONNX file: " + onnx_path.string());
    }
    env->alg = std::make_unique<isaaclab::OrtRunner>(onnx_path);

    if (debug_print)
    {
        spdlog::info("State_PoseTracking initialized with policy from {}", policy_dir.string());
        spdlog::info("Goal pose topic: {}", goal_pose_topic);
    }

    // Register safety check for bad orientation
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]() -> bool
            { return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")));
}

void State_PoseTracking::run()
{
    // Debug output
    if (debug_print && goal_pose_sub)
    {
        static int print_counter = 0;
        if (print_counter++ % 50 == 0) // Print every 50 steps (1 second at 50Hz or 0.5 seconds at 100Hz)
        {
            auto goal_pose = goal_pose_sub->getData();
            if (goal_pose.is_valid)
            {
                spdlog::info("Goal pose: x={:.3f}, y={:.3f}, theta={:.3f}",
                             goal_pose.x, goal_pose.y, goal_pose.theta);
            }
        }
    }

    auto action = env->action_manager->processed_actions();
    for (int i(0); i < env->robot->data.joint_ids_map.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
