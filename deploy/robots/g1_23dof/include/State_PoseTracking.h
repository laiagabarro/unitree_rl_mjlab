// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "goal_pose.h"

class State_PoseTracking : public FSMState
{
public:
    State_PoseTracking(int state_mode, std::string state_string);

    void enter()
    {
        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        env->robot->update();

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
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    isaaclab::GoalPoseSubscription::SharedPtr goal_pose_sub;

    std::thread policy_thread;
    bool policy_thread_running = false;

    // Debug flag
    bool debug_print = true;
};

REGISTER_FSM(State_PoseTracking)
