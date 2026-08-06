#include "State_PosePID.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <cmath>

State_PosePID::State_PosePID(int state_mode, std::string state_string)
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

    // Create environment
    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
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

    // Load PID gains from config
    try
    {
        auto pid_cfg = cfg["pid_gains"];
        if (pid_cfg)
        {
            kp_x = pid_cfg["kp_x"].as<float>();
            kp_y = pid_cfg["kp_y"].as<float>();
            kp_yaw = pid_cfg["kp_yaw"].as<float>();
            spdlog::info("PID gains: kp_x={}, kp_y={}, kp_yaw={}", kp_x, kp_y, kp_yaw);
        }
    }
    catch (const std::exception &e)
    {
        spdlog::info("Using default PID gains: kp_x={}, kp_y={}, kp_yaw={}", kp_x, kp_y, kp_yaw);
    }

    // Load velocity limits from config or use environment config
    try
    {
        if (cfg["velocity_limits"].IsDefined())
        {
            auto vel_cfg = cfg["velocity_limits"];
            max_vel_x = vel_cfg["max_vel_x"].as<float>(max_vel_x);
            min_vel_x = vel_cfg["min_vel_x"].as<float>(min_vel_x);
            max_vel_y = vel_cfg["max_vel_y"].as<float>(max_vel_y);
            min_vel_y = vel_cfg["min_vel_y"].as<float>(min_vel_y);
            max_vel_yaw = vel_cfg["max_vel_yaw"].as<float>(max_vel_yaw);
            min_vel_yaw = vel_cfg["min_vel_yaw"].as<float>(min_vel_yaw);
            spdlog::info("Velocity limits from config: x=[{}, {}], y=[{}, {}], yaw=[{}, {}]",
                         min_vel_x, max_vel_x, min_vel_y, max_vel_y, min_vel_yaw, max_vel_yaw);
        }
        else if (env->cfg["commands"]["base_velocity"]["ranges"].IsDefined())
        {
            auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];
            min_vel_x = ranges["lin_vel_x"][0].as<float>();
            max_vel_x = ranges["lin_vel_x"][1].as<float>();
            min_vel_y = ranges["lin_vel_y"][0].as<float>();
            max_vel_y = ranges["lin_vel_y"][1].as<float>();
            min_vel_yaw = ranges["ang_vel_z"][0].as<float>();
            max_vel_yaw = ranges["ang_vel_z"][1].as<float>();
            spdlog::info("Velocity limits: x=[{}, {}], y=[{}, {}], yaw=[{}, {}]",
                         min_vel_x, max_vel_x, min_vel_y, max_vel_y, min_vel_yaw, max_vel_yaw);
        }
    }
    catch (const std::exception &e)
    {
        spdlog::warn("Could not load velocity limits from config, using defaults");
    }

    // Enable debug prints
    try
    {
        debug_print = cfg["debug_print"].as<bool>();
    }
    catch (const std::exception &e)
    {
        debug_print = true; // Default to true for debugging
    }

    // Register safety check for bad orientation
    this->registered_checks.emplace_back(
        std::make_pair(
            [&]() -> bool
            { return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")));
}

void State_PosePID::run()
{
    auto action = env->action_manager->processed_actions();
    for (int i(0); i < env->robot->data.joint_ids_map.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}

std::vector<float> State_PosePID::compute_velocity_commands()
{
    std::vector<float> vel_cmd(3, 0.0f);

    if (!goal_pose_sub)
    {
        return vel_cmd;
    }

    auto goal_pose = goal_pose_sub->getData();

    if (!goal_pose.is_valid)
    {
        if (debug_print)
        {
            static int warn_counter = 0;
            if (warn_counter++ % 100 == 0)
            {
                spdlog::warn("Goal pose not valid yet");
            }
        }
        return vel_cmd;
    }

    // Goal pose is in robot's base frame
    // Positive x is forward, positive y is left, positive theta is counter-clockwise
    float error_x = goal_pose.x;
    float error_y = goal_pose.y;
    float error_theta = goal_pose.theta;

    float norm = std::sqrt(error_x * error_x + error_y * error_y);
    if (norm > 1.0e-6f)
    {
        float min_safe_distance = 0.0f; // 1 m
        if (norm < min_safe_distance)
        {
            error_x = -min_safe_distance * (error_x / norm);
            error_y = -min_safe_distance * (error_y / norm);
            if (debug_print)            {
                spdlog::info("Applying min_safe_distance: norm={:.3f}, error_x={:.3f}, error_y={:.3f}", norm, error_x, error_y);
            }
        }
    }

    // Normalize theta error to [-pi, pi]
    while (error_theta > M_PI)
        error_theta -= 2.0f * M_PI;
    while (error_theta < -M_PI)
        error_theta += 2.0f * M_PI;

    // Proportional control
    vel_cmd[0] = kp_x * error_x;
    vel_cmd[1] = kp_y * error_y;
    vel_cmd[2] = kp_yaw * error_theta;

    // Higher scaling for -x direction
    if (vel_cmd[0] < 0)
    {
        vel_cmd[0] *= 2.0f;
    }

    // Clamp to velocity limits
    vel_cmd[0] = std::clamp(vel_cmd[0], min_vel_x, max_vel_x);
    vel_cmd[1] = std::clamp(vel_cmd[1], min_vel_y, max_vel_y);
    vel_cmd[2] = std::clamp(vel_cmd[2], min_vel_yaw, max_vel_yaw);

    // Debug output
    if (debug_print)
    {
        static int print_counter = 0;
        if (print_counter++ % 50 == 0) // Print every 50 steps (0.5 seconds at 100Hz)
        {
            spdlog::info("Goal: x={:.3f}, y={:.3f}, theta={:.3f} | Vel cmd: x={:.3f}, y={:.3f}, yaw={:.3f}",
                         error_x, error_y, error_theta, vel_cmd[0], vel_cmd[1], vel_cmd[2]);
        }
    }

    return vel_cmd;
}
