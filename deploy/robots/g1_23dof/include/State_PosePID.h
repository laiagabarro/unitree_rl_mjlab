// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "goal_pose.h"

class State_PosePID : public FSMState
{
public:
    State_PosePID(int state_mode, std::string state_string);

    void enter()
    {
        FSMState::enter();
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();
        
        // Initialize command velocity and set pointer in articulation data
        command_velocity.setZero();
        env->robot->data.command_velocity = &command_velocity;

        // Wait for goal pose connection
        if (goal_pose_sub)
        {
            spdlog::info("Waiting for goal pose connection...");
            goal_pose_sub->wait_for_connection();
            spdlog::info("Goal pose connected!");
        }

        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]
                                    {
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                // Compute velocity commands from goal pose and store in command_velocity
                auto vel_cmd = compute_velocity_commands();
                command_velocity[0] = vel_cmd[0];  // vel_x
                command_velocity[1] = vel_cmd[1];  // vel_y
                command_velocity[2] = vel_cmd[2];  // vel_yaw
                
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            } });
    }

    void run();

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable())
        {
            policy_thread.join();
        }
        
        // Clear command velocity pointer
        if (env->robot->data.command_velocity == &command_velocity)
        {
            env->robot->data.command_velocity = nullptr;
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    isaaclab::GoalPoseSubscription::SharedPtr goal_pose_sub;

    std::thread policy_thread;
    bool policy_thread_running = false;
    
    // Command velocity storage (referenced by env->robot->data.command_velocity)
    Eigen::Vector3f command_velocity;

    // PID gains (starting with just P control)
    float kp_x = 1.0f;
    float kp_y = 1.0f;
    float kp_yaw = 1.0f;

    // Velocity limits (will be loaded from config)
    float max_vel_x = 0.5f;
    float min_vel_x = -0.5f;
    float max_vel_y = 0.5f;
    float min_vel_y = -0.5f;
    float max_vel_yaw = 1.0f;
    float min_vel_yaw = -1.0f;

    // Debug flag
    bool debug_print = true;

    /**
     * @brief Compute velocity commands using PID control
     * @return Vector of [vel_x, vel_y, vel_yaw]
     */
    std::vector<float> compute_velocity_commands();
};

REGISTER_FSM(State_PosePID)
